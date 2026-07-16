"""
Phase 5 (final): author-grouped cross-validation
5-fold GroupKFold (no author appears in both train and test of the same fold)
Reports accuracy/F1 as mean +/- standard deviation across folds, and
McNemar's test over the pooled out-of-fold predictions (each sample
evaluated exactly once, in the fold where it was held out).

Note: random seeds are fixed for NumPy, scikit-learn, and PyTorch. Without
this, LSTM training is not deterministic across runs (weight initialization
and minibatch shuffling) -- see Section 4.3 of the paper.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 5
N_FOLDS = 5
COMMIT_FEATURES = ['deletion_ratio', 'hours_from_own_baseline', 'is_weekend_num', 'log_churn', 'log_iat']

df = pd.read_parquet('data/processed/github_features_final.parquet')
win_df = pd.read_parquet('data/processed/github_windows_groundtruth.parquet')

df['is_weekend_num'] = df['is_weekend'].astype(int)
df['log_churn'] = np.log1p(df['total_churn'])
df['log_iat'] = np.log1p(df['iat_hours'].fillna(df.groupby('author_id')['iat_hours'].transform('median')).fillna(0))
df = df.sort_values(['author_id', 'commit_seq_number']).reset_index(drop=True)

label_lookup = win_df.set_index(['author_id', 'window_end_seq'])['iso_anomaly'].to_dict()

# ==========================================
# BUILD ORACLE PAIRS: (raw sequence of W_t, label of W_{t+1})
# ==========================================
X_list, y_list, group_list = [], [], []
for author_id, grp in df.groupby('author_id'):
    grp = grp.reset_index(drop=True)
    n = len(grp)
    if n < SEQ_LEN + 1:
        continue
    feat_matrix = grp[COMMIT_FEATURES].values
    seq_numbers = grp['commit_seq_number'].values
    for start in range(0, n - SEQ_LEN):
        window_end_idx = start + SEQ_LEN - 1
        next_idx = window_end_idx + 1
        if next_idx >= n:
            continue
        label = label_lookup.get((author_id, seq_numbers[next_idx]))
        if label is None:
            continue
        X_list.append(feat_matrix[start:start + SEQ_LEN])
        y_list.append(int(label))
        group_list.append(author_id)

X = np.array(X_list)
y = np.array(y_list)
groups = np.array(group_list)
print(f"Total oracle sequences: {len(X)} | Authors: {len(set(groups))} | Positive balance: {y.mean()*100:.1f}%")


class OracleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gkf = GroupKFold(n_splits=N_FOLDS)
n_feat = X.shape[2]

fold_results = []
oof_rf_preds = np.full(len(y), -1)
oof_lstm_preds = np.full(len(y), -1)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, "Author leakage between folds!"

    # Scaling fit ONLY on train, to avoid leakage into test
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train_s = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
    rf.fit(X_train_s.reshape(len(X_train_s), -1), y_train)
    rf_preds = rf.predict(X_test_s.reshape(len(X_test_s), -1))
    oof_rf_preds[test_idx] = rf_preds

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average='binary', zero_division=0)

    # --- LSTM ---
    model = OracleLSTM(input_size=n_feat).to(device)
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(len(y_train) / (2.0 * class_counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_train = len(X_train_t)
    for epoch in range(30):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 64):
            idx = perm[i:i+64]
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        lstm_preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()
    oof_lstm_preds[test_idx] = lstm_preds

    lstm_acc = accuracy_score(y_test, lstm_preds)
    lstm_p, lstm_r, lstm_f1, _ = precision_recall_fscore_support(y_test, lstm_preds, average='binary', zero_division=0)

    print(f"  RF   -> acc={rf_acc:.4f} prec={rf_p:.4f} rec={rf_r:.4f} f1={rf_f1:.4f}")
    print(f"  LSTM -> acc={lstm_acc:.4f} prec={lstm_p:.4f} rec={lstm_r:.4f} f1={lstm_f1:.4f}")

    fold_results.append({
        'fold': fold+1, 'n_test': len(y_test), 'n_authors_test': len(set(groups[test_idx])),
        'rf_acc': rf_acc, 'rf_precision': rf_p, 'rf_recall': rf_r, 'rf_f1': rf_f1,
        'lstm_acc': lstm_acc, 'lstm_precision': lstm_p, 'lstm_recall': lstm_r, 'lstm_f1': lstm_f1,
    })

results_df = pd.DataFrame(fold_results)
print("\n=== RESULTS PER FOLD ===")
print(results_df.to_string(index=False))

print("\n=== MEAN +/- STANDARD DEVIATION ACROSS FOLDS ===")
for model_name in ['rf', 'lstm']:
    for metric in ['acc', 'precision', 'recall', 'f1']:
        col = f'{model_name}_{metric}'
        print(f"  {col}: {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")

# ==========================================
# McNEMAR OVER POOLED OUT-OF-FOLD PREDICTIONS (each sample evaluated exactly once)
# ==========================================
assert (oof_rf_preds >= 0).all() and (oof_lstm_preds >= 0).all(), "Missing out-of-fold predictions"

rf_correct = (oof_rf_preds == y)
lstm_correct = (oof_lstm_preds == y)
both = np.sum(rf_correct & lstm_correct)
only_rf = np.sum(rf_correct & ~lstm_correct)
only_lstm = np.sum(~rf_correct & lstm_correct)
neither = np.sum(~rf_correct & ~lstm_correct)

table = [[both, only_rf], [only_lstm, neither]]
print(f"\n=== McNemar over pooled out-of-fold predictions (n={len(y)}) ===")
print("Table:", table)
mc = mcnemar(table, exact=False, correction=True)
print(f"McNemar statistic: {mc.statistic:.4f}, p-value: {mc.pvalue:.6f}")

print(f"\nGlobal OOF accuracy -> RF: {accuracy_score(y, oof_rf_preds):.4f} | LSTM: {accuracy_score(y, oof_lstm_preds):.4f}")

results_df.to_csv('data/processed/fase5_cv_resultados.csv', index=False)
np.savez('data/processed/fase5_oof_predictions.npz',
         y_true=y, oof_rf=oof_rf_preds, oof_lstm=oof_lstm_preds)
trace5 = {
    'n_folds': N_FOLDS,
    'rf_acc_mean': results_df['rf_acc'].mean(), 'rf_acc_std': results_df['rf_acc'].std(),
    'lstm_acc_mean': results_df['lstm_acc'].mean(), 'lstm_acc_std': results_df['lstm_acc'].std(),
    'rf_f1_mean': results_df['rf_f1'].mean(), 'rf_f1_std': results_df['rf_f1'].std(),
    'lstm_f1_mean': results_df['lstm_f1'].mean(), 'lstm_f1_std': results_df['lstm_f1'].std(),
    'mcnemar_oof_statistic': mc.statistic, 'mcnemar_oof_pvalue': mc.pvalue,
    'oof_accuracy_rf': accuracy_score(y, oof_rf_preds), 'oof_accuracy_lstm': accuracy_score(y, oof_lstm_preds),
}
pd.Series(trace5).to_csv('data/processed/fase5_cv_trace.csv', header=['valor'])
print("\nSaved: fase5_cv_resultados.csv, fase5_cv_trace.csv")
