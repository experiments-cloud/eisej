"""
Phase 3 (final): comparison of ground truth generation methods
  A) Continuous risk index + percentile threshold (no unsupervised
     "discovery" claim)
  B) Anomaly detection designed to find genuine minorities: Isolation
     Forest and Gaussian Mixture Model
Reports which is more robust (real separation on external validation
variables, cross-method agreement, seed stability). This is the script
that produces the final ground truth used throughout the paper
(Isolation Forest, Section 3.4).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from scipy.stats import mannwhitneyu

win_df = pd.read_parquet('data/processed/github_windows_clustered.parquet')

# --- Fix: erratic_pacing must capture IRREGULARITY relative to the author's
#     own pace, not just IAT magnitude (which really reflects commit
#     frequency, a role/dedication trait, not necessarily risk).
#     Coefficient of variation is used instead. ---
win_df['iat_cv'] = win_df['iat_std'] / win_df['iat_mean'].replace(0, np.nan)
win_df['iat_cv'] = win_df['iat_cv'].fillna(win_df['iat_cv'].median())
z_cv = StandardScaler().fit_transform(np.log1p(win_df[['iat_cv']]))[:, 0]
win_df['erratic_pacing_index_v2'] = z_cv

FEATURES = ['operational_friction_index', 'boundary_dissolution_index', 'erratic_pacing_index_v2']
X = win_df[FEATURES].values
X_scaled = StandardScaler().fit_transform(X)

# ==========================================
# APPROACH A: CONTINUOUS RISK INDEX (percentiles)
# ==========================================
win_df['risk_score'] = X_scaled.mean(axis=1)

results = {}
for pct in [0.10, 0.15, 0.20]:
    threshold = win_df['risk_score'].quantile(1 - pct)
    win_df[f'high_risk_p{int(pct*100)}'] = win_df['risk_score'] >= threshold

print("=== APPROACH A: Continuous index + percentile threshold ===")
for pct in [10, 15, 20]:
    col = f'high_risk_p{pct}'
    grp_hi = win_df[win_df[col]]
    grp_lo = win_df[~win_df[col]]
    u, p = mannwhitneyu(grp_hi['is_fix_rate'], grp_lo['is_fix_rate'], alternative='two-sided')
    print(f"  p{pct}: n={grp_hi.shape[0]} ({grp_hi.shape[0]/len(win_df)*100:.1f}%) | "
          f"high fix_rate={grp_hi['is_fix_rate'].mean():.3f} vs low={grp_lo['is_fix_rate'].mean():.3f} | p={p:.2e}")
    results[f'A_p{pct}_n'] = grp_hi.shape[0]
    results[f'A_p{pct}_fixrate_hi'] = grp_hi['is_fix_rate'].mean()
    results[f'A_p{pct}_fixrate_lo'] = grp_lo['is_fix_rate'].mean()
    results[f'A_p{pct}_pvalue'] = p

# ==========================================
# APPROACH B1: ISOLATION FOREST
# ==========================================
print("\n=== APPROACH B1: Isolation Forest (contamination=0.15) ===")
iso = IsolationForest(contamination=0.15, random_state=42, n_estimators=200)
win_df['iso_anomaly'] = iso.fit_predict(X_scaled) == -1
n_iso = win_df['iso_anomaly'].sum()
print(f"  Anomalous detected: {n_iso} ({n_iso/len(win_df)*100:.1f}%)")

grp_iso_hi = win_df[win_df['iso_anomaly']]
grp_iso_lo = win_df[~win_df['iso_anomaly']]
u_iso, p_iso = mannwhitneyu(grp_iso_hi['is_fix_rate'], grp_iso_lo['is_fix_rate'], alternative='two-sided')
print(f"  fix_rate anomalous={grp_iso_hi['is_fix_rate'].mean():.3f} vs normal={grp_iso_lo['is_fix_rate'].mean():.3f} | p={p_iso:.2e}")

# Sensitivity: Isolation Forest membership stability across seeds.
# NOTE: comparing only the PROPORTION flagged across seeds is trivially
# stable, since `contamination=0.15` forces that exact size regardless of
# the seed. The real robustness check is whether the SAME windows get
# flagged -- measured here via the Jaccard index of set membership between
# every pair of seeds (this is the 'jaccard_stability' figure reported in
# the paper's Table 5, J=0.867).
iso_seed_sets = []
for seed in [0, 1, 7, 42, 123]:
    iso_s = IsolationForest(contamination=0.15, random_state=seed, n_estimators=200)
    labels = iso_s.fit_predict(X_scaled) == -1
    iso_seed_sets.append(set(np.where(labels)[0]))

jaccard_pairs = []
for i in range(len(iso_seed_sets)):
    for j in range(i + 1, len(iso_seed_sets)):
        a, b = iso_seed_sets[i], iso_seed_sets[j]
        jaccard_pairs.append(len(a & b) / len(a | b))
jaccard_stability = np.mean(jaccard_pairs)
print(f"  Cross-seed membership stability (mean pairwise Jaccard): {jaccard_stability:.3f}")

# ==========================================
# APPROACH B2: GAUSSIAN MIXTURE MODEL (free covariance -> allows unequal cluster sizes)
# ==========================================
print("\n=== APPROACH B2: Gaussian Mixture Model (k=2, covariance_type='full') ===")
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=10)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_sizes = pd.Series(gmm_labels).value_counts(normalize=True).sort_index()
print(f"  Proportions: {dict(gmm_sizes)}")
minority_gmm = gmm_sizes.idxmin()
win_df['gmm_anomaly'] = gmm_labels == minority_gmm
n_gmm = win_df['gmm_anomaly'].sum()
print(f"  Minority cluster: {n_gmm} ({n_gmm/len(win_df)*100:.1f}%)")

grp_gmm_hi = win_df[win_df['gmm_anomaly']]
grp_gmm_lo = win_df[~win_df['gmm_anomaly']]
u_gmm, p_gmm = mannwhitneyu(grp_gmm_hi['is_fix_rate'], grp_gmm_lo['is_fix_rate'], alternative='two-sided')
print(f"  fix_rate minority={grp_gmm_hi['is_fix_rate'].mean():.3f} vs majority={grp_gmm_lo['is_fix_rate'].mean():.3f} | p={p_gmm:.2e}")

# ==========================================
# CROSS-METHOD AGREEMENT (convergent validity)
# ==========================================
print("\n=== Cross-method agreement (do they flag the same authors/windows?) ===")
def jaccard(a, b):
    a, b = set(a[a].index), set(b[b].index)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)

j_A15_iso = jaccard(win_df['high_risk_p15'], win_df['iso_anomaly'])
j_A15_gmm = jaccard(win_df['high_risk_p15'], win_df['gmm_anomaly'])
j_iso_gmm = jaccard(win_df['iso_anomaly'], win_df['gmm_anomaly'])
print(f"  Jaccard (percentile15 vs IsolationForest): {j_A15_iso:.3f}")
print(f"  Jaccard (percentile15 vs GMM): {j_A15_gmm:.3f}")
print(f"  Jaccard (IsolationForest vs GMM): {j_iso_gmm:.3f}")

# ==========================================
# FINAL COMPARISON TABLE
# ==========================================
print("\n=== ROBUSTNESS COMPARISON TABLE ===")
comparison = pd.DataFrame({
    'method': ['Percentile 15%', 'Isolation Forest', 'GMM (k=2, full cov.)'],
    'proportion_pct': [
        win_df['high_risk_p15'].mean() * 100,
        win_df['iso_anomaly'].mean() * 100,
        win_df['gmm_anomaly'].mean() * 100,
    ],
    'fix_rate_gap': [
        grp_hi['is_fix_rate'].mean() - grp_lo['is_fix_rate'].mean(),
        grp_iso_hi['is_fix_rate'].mean() - grp_iso_lo['is_fix_rate'].mean(),
        grp_gmm_hi['is_fix_rate'].mean() - grp_gmm_lo['is_fix_rate'].mean(),
    ],
    'p_value': [results['A_p15_pvalue'], p_iso, p_gmm],
    'jaccard_stability': [np.nan, jaccard_stability, np.nan],
})
print(comparison.to_string(index=False))

# ==========================================
# SAVE
# ==========================================
# Isolation Forest (approach B1) is selected as the final ground truth
# method for the paper -- see Table "groundtruth" in Section 3.4.
win_df.to_parquet('data/processed/github_windows_groundtruth.parquet', index=False)
comparison.to_csv('data/processed/groundtruth_method_comparison.csv', index=False)
pd.Series(results).to_csv('data/processed/fase3_trace_enfoqueA.csv', header=['valor'])
print("\nSaved: github_windows_groundtruth.parquet, groundtruth_method_comparison.csv, fase3_trace_enfoqueA.csv")
