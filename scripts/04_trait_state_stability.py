"""
Phase 4: Trait vs. State
For each author with enough windows, evaluates whether their anomalous
windows (Isolation Forest) are (a) persistent and spread across their whole
history -> STABLE TRAIT, or (b) concentrated in a bounded time episode ->
EMERGENT STATE (a defensible early-warning signal).

Note: the 'clasificacion' column values ('rasgo_estable', 'estado_emergente',
'mixto') are kept in Spanish to stay consistent with the already-released
data/processed/fase4_clasificacion_rasgo_estado.csv file. They correspond to
"stable_trait" / "emergent_state" / "mixed" as used in the paper text.
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy

MIN_WINDOWS_FOR_PROFILE = 10   # minimum windows per author for temporal profiling
win_df = pd.read_parquet('data/processed/github_windows_groundtruth.parquet')
win_df = win_df.sort_values(['author_id', 'window_end_seq']).reset_index(drop=True)

author_counts = win_df.groupby('author_id').size()
eligible_authors = author_counts[author_counts >= MIN_WINDOWS_FOR_PROFILE].index
print(f"Authors with >= {MIN_WINDOWS_FOR_PROFILE} windows (eligible for temporal profiling): {len(eligible_authors)} / {win_df['author_id'].nunique()}")

profiles = []
for author_id in eligible_authors:
    a = win_df[win_df['author_id'] == author_id].sort_values('window_end_seq').reset_index(drop=True)
    n = len(a)
    anomaly_rate = a['iso_anomaly'].mean()
    n_anomalous = a['iso_anomaly'].sum()

    # --- Temporal concentration: split the author's history into 4 chronological quartiles ---
    a['quartile'] = pd.qcut(a.index, 4, labels=False, duplicates='drop')
    q_counts = a.groupby('quartile')['iso_anomaly'].sum()
    q_counts = q_counts.reindex(range(4), fill_value=0)

    if n_anomalous > 0:
        q_probs = q_counts / n_anomalous
        q_probs_nonzero = q_probs[q_probs > 0]
        ent = shannon_entropy(q_probs_nonzero, base=2)
        ent_norm = ent / np.log2(4)  # 0 = concentrated in 1 quartile, 1 = perfectly spread out
    else:
        ent_norm = np.nan

    # --- Lag-1 autocorrelation of the continuous risk_score (behavioral persistence) ---
    if n >= 3:
        s = a['risk_score'].values
        autocorr_lag1 = np.corrcoef(s[:-1], s[1:])[0, 1]
    else:
        autocorr_lag1 = np.nan

    profiles.append({
        'author_id': author_id,
        'n_windows': n,
        'n_anomalous': n_anomalous,
        'anomaly_rate': anomaly_rate,
        'temporal_entropy_norm': ent_norm,
        'autocorr_risk_lag1': autocorr_lag1,
    })

prof_df = pd.DataFrame(profiles)

# --- Trait vs. state classification ---
# Only authors with AT LEAST 1 anomalous window are classified (those who never
# had one simply don't apply to the trait/state scheme, they remain "no signal")
has_signal = prof_df[prof_df['n_anomalous'] >= 2].copy()  # min 2 so entropy is meaningful

rate_median = has_signal['anomaly_rate'].median()
entropy_median = has_signal['temporal_entropy_norm'].median()

def classify(row):
    high_rate = row['anomaly_rate'] >= rate_median
    high_entropy = row['temporal_entropy_norm'] >= entropy_median
    if high_rate and high_entropy:
        return 'rasgo_estable'       # persistent and spread out in time -> "stable trait"
    elif not high_rate and not high_entropy:
        return 'estado_emergente'    # infrequent and concentrated in one episode -> "emergent state"
    else:
        return 'mixto'               # "mixed"

has_signal['clasificacion'] = has_signal.apply(classify, axis=1)

print("\n=== Classification thresholds (medians) ===")
print(f"anomaly_rate median: {rate_median:.3f}")
print(f"temporal_entropy_norm median: {entropy_median:.3f}")

print("\n=== Classification distribution (authors with >=2 anomalous windows) ===")
print(has_signal['clasificacion'].value_counts())
print(has_signal['clasificacion'].value_counts(normalize=True) * 100)

n_sin_senal = (prof_df['n_anomalous'] < 2).sum()
print(f"\nAuthors without sufficient signal (<2 anomalous windows): {n_sin_senal} / {len(prof_df)}")

print("\n=== Lag-1 autocorrelation of the continuous risk_score (all eligible authors) ===")
print(prof_df['autocorr_risk_lag1'].describe())
print(f"\n% of authors with significant positive autocorrelation (>0.2): "
      f"{(prof_df['autocorr_risk_lag1'] > 0.2).mean() * 100:.1f}%")

# ==========================================
# METHODOLOGICAL CORRECTION: lag-5 (non-overlapping) autocorrelation
# ==========================================
# Windows are built with step=1 (see 03_groundtruth_windows.py), so two
# consecutive windows share 4 of their 5 commits. This mechanically inflates
# lag-1 autocorrelation and does NOT constitute valid evidence of behavioral
# persistence. Lag-5 compares windows that share zero commits, and is the
# measure actually used for the paper's trait/state conclusion (Table 8,
# Section 4.2).
LAG = 5
autocorrs_lag5 = []
for author_id in eligible_authors:
    a = win_df[win_df['author_id'] == author_id].sort_values('window_end_seq').reset_index(drop=True)
    s = a['risk_score'].values
    if len(s) >= LAG + 2:
        autocorrs_lag5.append(np.corrcoef(s[:-LAG], s[LAG:])[0, 1])

print(f"\n=== Lag-5 autocorrelation (non-overlapping windows), n={len(autocorrs_lag5)} ===")
print(f"Median: {np.median(autocorrs_lag5):.3f}  |  % > 0.2: {(np.array(autocorrs_lag5) > 0.2).mean()*100:.1f}%")
print("This is the corrected figure reported in Table 8 of the paper; the")
print("lag-1 figure above is reported alongside it specifically to make the")
print("magnitude of the window-overlap artifact visible (Section 4.2).")

print("\n=== Average profile by classification ===")
print(has_signal.groupby('clasificacion')[['n_windows', 'anomaly_rate', 'temporal_entropy_norm', 'autocorr_risk_lag1']].mean())

# ==========================================
# SAVE
# ==========================================
prof_df.to_parquet('data/processed/fase4_author_profiles.parquet', index=False)
has_signal.to_csv('data/processed/fase4_clasificacion_rasgo_estado.csv', index=False)

trace4 = {
    'autores_elegibles_perfil': len(eligible_authors),
    'autores_con_senal_suficiente': len(has_signal),
    'autores_sin_senal': n_sin_senal,
    'pct_rasgo_estable': (has_signal['clasificacion'] == 'rasgo_estable').mean() * 100,
    'pct_estado_emergente': (has_signal['clasificacion'] == 'estado_emergente').mean() * 100,
    'pct_mixto': (has_signal['clasificacion'] == 'mixto').mean() * 100,
    'autocorr_lag1_mediana': prof_df['autocorr_risk_lag1'].median(),
    'pct_autocorr_positiva_significativa': (prof_df['autocorr_risk_lag1'] > 0.2).mean() * 100,
}
pd.Series(trace4).to_csv('data/processed/fase4_trace.csv', header=['valor'])
print("\nSaved: fase4_author_profiles.parquet, fase4_clasificacion_rasgo_estado.csv, fase4_trace.csv")
