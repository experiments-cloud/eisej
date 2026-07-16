"""
Fase 5: Modelado predictivo oracle
Input: secuencia cruda de 5 commits (ventana W_t)
Target: estado (Isolation Forest) de la SIGUIENTE ventana W_{t+1}
Split: GroupShuffleSplit por autor (85/15) -> ningún autor aparece en ambos splits
Modelos: LSTM (temporal) vs Random Forest (estático, secuencia aplanada)
Validación: McNemar's test sobre las clasificaciones del hold-out
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

SEQ_LEN = 5
COMMIT_FEATURES = ['deletion_ratio', 'hours_from_own_baseline', 'is_weekend_num', 'log_churn', 'log_iat']

df = pd.read_parquet('data/github_features_final.parquet')
win_df = pd.read_parquet('data/github_windows_groundtruth.parquet')

df['is_weekend_num'] = df['is_weekend'].astype(int)
df['log_churn'] = np.log1p(df['total_churn'])
df['log_iat'] = np.log1p(df['iat_hours'].fillna(df.groupby('author_id')['iat_hours'].transform('median')).fillna(0))
df = df.sort_values(['author_id', 'commit_seq_number']).reset_index(drop=True)

# Diccionario rápido de etiquetas por (author_id, window_end_seq)
label_lookup = win_df.set_index(['author_id', 'window_end_seq'])['iso_anomaly'].to_dict()

# ==========================================
# CONSTRUCCIÓN DE PARES ORACLE: (secuencia cruda de W_t, etiqueta de W_{t+1})
# ==========================================
X_list, y_list, group_list = [], [], []

for author_id, grp in df.groupby('author_id'):
    grp = grp.reset_index(drop=True)
    n = len(grp)
    if n < SEQ_LEN + 1:
        continue
    feat_matrix = grp[COMMIT_FEATURES].values  # (n, num_features)
    seq_numbers = grp['commit_seq_number'].values

    # ventana W_t termina en la posición idx (0-indexed: start..start+SEQ_LEN-1)
    for start in range(0, n - SEQ_LEN):  # deja al menos 1 commit extra para W_{t+1}
        window_end_idx = start + SEQ_LEN - 1
        next_window_end_idx = window_end_idx + 1
        if next_window_end_idx >= n:
            continue

        window_end_seq_t = seq_numbers[window_end_idx]
        window_end_seq_t1 = seq_numbers[next_window_end_idx]

        label = label_lookup.get((author_id, window_end_seq_t1))
        if label is None:
            continue

        X_list.append(feat_matrix[start:start + SEQ_LEN])
        y_list.append(int(label))
        group_list.append(author_id)

X = np.array(X_list)               # (N, 5, num_features)
y = np.array(y_list)                # (N,)
groups = np.array(group_list)

print(f"Total secuencias oracle construidas: {len(X)}")
print(f"Autores representados: {len(set(groups))}")
print(f"Balance de clases: {y.mean()*100:.1f}% positivas (ventana siguiente anómala)")

# ==========================================
# SPLIT POR AUTOR (GroupShuffleSplit, evita fuga de datos)
# ==========================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

overlap = set(groups_train) & set(groups_test)
print(f"\nAutores en train: {len(set(groups_train))} | Autores en test: {len(set(groups_test))}")
print(f"Overlap de autores entre train/test (debe ser 0): {len(overlap)}")
print(f"Train: {len(X_train)} secuencias ({y_train.mean()*100:.1f}% positivas)")
print(f"Test:  {len(X_test)} secuencias ({y_test.mean()*100:.1f}% positivas)")

# ==========================================
# ESCALADO (fit SOLO en train, evita fuga de información)
# ==========================================
n_feat = X_train.shape[2]
scaler = StandardScaler()
X_train_flat_for_scaler = X_train.reshape(-1, n_feat)
scaler.fit(X_train_flat_for_scaler)

def scale_seq(Xarr):
    shape = Xarr.shape
    flat = Xarr.reshape(-1, n_feat)
    scaled = scaler.transform(flat)
    return scaled.reshape(shape)

X_train_s = scale_seq(X_train)
X_test_s = scale_seq(X_test)

# ==========================================
# MODELO 1: RANDOM FOREST (estático, secuencia aplanada)
# ==========================================
print("\n=== Entrenando Random Forest ===")
X_train_flat = X_train_s.reshape(len(X_train_s), -1)
X_test_flat = X_test_s.reshape(len(X_test_s), -1)

rf = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
rf.fit(X_train_flat, y_train)
rf_preds = rf.predict(X_test_flat)

rf_acc = accuracy_score(y_test, rf_preds)
rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average='binary', zero_division=0)
print(f"RF -> Accuracy: {rf_acc:.4f} | Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f} | F1: {rf_f1:.4f}")
print("Matriz de confusión RF:\n", confusion_matrix(y_test, rf_preds))

# ==========================================
# MODELO 2: LSTM (temporal)
# ==========================================
print("\n=== Entrenando LSTM ===")

class OracleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OracleLSTM(input_size=n_feat).to(device)

X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# Pesos de clase dinámicos (igual que Random Forest balanced)
class_counts = np.bincount(y_train)
class_weights = torch.tensor(len(y_train) / (2.0 * class_counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

BATCH_SIZE = 64
N_EPOCHS = 50
n_train = len(X_train_t)

for epoch in range(N_EPOCHS):
    model.train()
    perm = torch.randperm(n_train)
    total_loss = 0
    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb, yb = X_train_t[idx], y_train_t[idx]
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / n_train
        print(f"  Epoch {epoch+1}/{N_EPOCHS} - loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    lstm_preds = torch.argmax(logits, dim=1).cpu().numpy()

lstm_acc = accuracy_score(y_test, lstm_preds)
lstm_prec, lstm_rec, lstm_f1, _ = precision_recall_fscore_support(y_test, lstm_preds, average='binary', zero_division=0)
print(f"\nLSTM -> Accuracy: {lstm_acc:.4f} | Precision: {lstm_prec:.4f} | Recall: {lstm_rec:.4f} | F1: {lstm_f1:.4f}")
print("Matriz de confusión LSTM:\n", confusion_matrix(y_test, lstm_preds))

# ==========================================
# McNEMAR'S TEST
# ==========================================
print("\n=== McNemar's Test (RF vs LSTM) ===")
rf_correct = (rf_preds == y_test)
lstm_correct = (lstm_preds == y_test)

both_correct = np.sum(rf_correct & lstm_correct)
only_rf = np.sum(rf_correct & ~lstm_correct)
only_lstm = np.sum(~rf_correct & lstm_correct)
both_wrong = np.sum(~rf_correct & ~lstm_correct)

table = [[both_correct, only_rf], [only_lstm, both_wrong]]
print("Tabla de contingencia [[ambos_correctos, solo_RF], [solo_LSTM, ambos_incorrectos]]:")
print(table)

result = mcnemar(table, exact=False, correction=True)
print(f"\nMcNemar statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")

# ==========================================
# GUARDADO
# ==========================================
summary = pd.DataFrame({
    'modelo': ['Random Forest', 'LSTM'],
    'accuracy': [rf_acc, lstm_acc],
    'precision': [rf_prec, lstm_prec],
    'recall': [rf_rec, lstm_rec],
    'f1': [rf_f1, lstm_f1],
})
summary.to_csv('data/fase5_resultados_modelos.csv', index=False)

trace5 = {
    'n_secuencias_totales': len(X),
    'n_train': len(X_train), 'n_test': len(X_test),
    'autores_train': len(set(groups_train)), 'autores_test': len(set(groups_test)),
    'overlap_autores_train_test': len(overlap),
    'balance_positivo_train': y_train.mean(), 'balance_positivo_test': y_test.mean(),
    'rf_accuracy': rf_acc, 'rf_f1': rf_f1,
    'lstm_accuracy': lstm_acc, 'lstm_f1': lstm_f1,
    'mcnemar_statistic': result.statistic, 'mcnemar_pvalue': result.pvalue,
}
pd.Series(trace5).to_csv('data/fase5_trace.csv', header=['valor'])
print("\n✅ Guardado: fase5_resultados_modelos.csv, fase5_trace.csv")
