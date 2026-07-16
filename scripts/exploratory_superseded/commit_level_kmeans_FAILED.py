"""
SUPERSEDED / NOT part of the final pipeline -- kept for transparency.

First ground truth attempt: k-means clustering directly on individual
commits (instead of 5-commit windows). Discarded for two reasons,
documented in Section 3.4 of the paper:
  1. Standardizing an imbalanced binary variable (is_weekend) alongside
     continuous variables let it dominate Euclidean distance, so the
     resulting clusters trivially reproduced the weekend/weekday split.
  2. Even after fixing that (via the composite indices below), clustering
     on individual commits did not reveal a differentiated minority: the
     resulting split was close to even (49.1% / 50.9%), consistent with
     the phenomenon being a sustained pattern over a sequence of
     contributions, not a property of a single isolated event.

Superseded by 03_groundtruth_windows.py (window-level analysis) and
03b_groundtruth_compare_methods.py (Isolation Forest selection).

Phase 3 (original): behavioral clustering (unsupervised ground truth)
- k selection (silhouette, k=2..6)
- Final k=2 clustering (theoretical framing: stable vs. anomalous profile)
- External validation: Defect Fix Rate / Revert Rate (chi-square), variables
  deliberately excluded from the clustering feature space
- Sensitivity analysis: stability of the minority cluster proportion
  across different seeds and under k=3
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency

df = pd.read_parquet('data/processed/github_features_final.parquet')

# --- Prepare the behavioral feature matrix ---
# is_fix / is_revert are deliberately excluded: reserved for EXTERNAL
# validation, they must not influence cluster formation (avoids trivial
# circularity).
work = df.copy()
work['iat_hours'] = work.groupby('author_id')['iat_hours'].transform(lambda s: s.fillna(s.median()))
work['iat_hours'] = work['iat_hours'].fillna(work['iat_hours'].median())
work['log_iat'] = np.log1p(work['iat_hours'].clip(lower=0))
work['is_weekend_num'] = work['is_weekend'].astype(int)

# --- Composition of 2 theoretical indices (avoids an imbalanced binary
#     variable dominating Euclidean distance when mixed raw with
#     continuous variables) ---
scaler_pre = StandardScaler()
z_hours = scaler_pre.fit_transform(work[['hours_from_own_baseline']])[:, 0]
z_weekend = scaler_pre.fit_transform(work[['is_weekend_num']])[:, 0]
work['boundary_dissolution_index'] = (z_hours + z_weekend) / 2

z_del = StandardScaler().fit_transform(work[['deletion_ratio']])[:, 0]
z_churn = StandardScaler().fit_transform(np.log1p(work[['total_churn']]))[:, 0]
work['operational_friction_index'] = (z_del + z_churn) / 2

FEATURES = ['operational_friction_index', 'boundary_dissolution_index', 'log_iat']
X = work[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== k selection (silhouette score, k=2..6) ===")
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores[k] = sil
    print(f"k={k}: silhouette={sil:.4f}")

# --- Final clustering: k=2, theoretical framing (stable vs. anomalous) ---
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
work['cluster'] = km2.fit_predict(X_scaled)

cluster_sizes = work['cluster'].value_counts(normalize=True).sort_index()
print("\n=== Cluster distribution (k=2) ===")
print(cluster_sizes)

# Identify minority cluster = "anomalous" (higher destructive churn + higher hour deviation)
cluster_profile = work.groupby('cluster')[['deletion_ratio', 'hours_from_own_baseline', 'is_weekend_num', 'iat_hours',
                                            'operational_friction_index', 'boundary_dissolution_index']].mean()
print("\n=== Average profile per cluster ===")
print(cluster_profile)

minority_cluster = cluster_sizes.idxmin()
work['is_anomalous'] = work['cluster'] == minority_cluster
print(f"\nAnomalous cluster identified: {minority_cluster} ({cluster_sizes[minority_cluster]*100:.1f}% of the corpus)")

# ==========================================
# EXTERNAL VALIDATION: Defect Fix Rate / Revert Rate
# ==========================================
print("\n=== External validation (variables NOT used in clustering) ===")
ext = work.groupby('is_anomalous')[['is_fix', 'is_revert']].mean()
print(ext)

# Chi-square: is_fix vs. cluster
ct_fix = pd.crosstab(work['is_anomalous'], work['is_fix'])
chi2_fix, p_fix, _, _ = chi2_contingency(ct_fix)
print(f"\nChi-square Defect Fix Rate: chi2={chi2_fix:.2f}, p={p_fix:.6f}")

ct_revert = pd.crosstab(work['is_anomalous'], work['is_revert'])
chi2_revert, p_revert, _, _ = chi2_contingency(ct_revert)
print(f"Chi-square Revert Rate: chi2={chi2_revert:.2f}, p={p_revert:.6f}")

# ==========================================
# SENSITIVITY ANALYSIS
# ==========================================
print("\n=== Sensitivity analysis: minority proportion stability ===")

print("\n-- Variation across random seeds (k=2) --")
seed_results = []
for seed in [0, 1, 7, 42, 123, 2024, 99999]:
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    prop = min(np.mean(labels == 0), np.mean(labels == 1))
    seed_results.append(prop)
    print(f"  seed={seed}: minority cluster = {prop*100:.2f}%")
print(f"  Mean: {np.mean(seed_results)*100:.2f}%  |  Std. dev.: {np.std(seed_results)*100:.2f}pp")

print("\n-- k=3 (subdivision of the anomalous cluster) --")
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels3 = km3.fit_predict(X_scaled)
work['cluster_k3'] = labels3
print(pd.Series(labels3).value_counts(normalize=True).sort_index())
print(work.groupby('cluster_k3')[['deletion_ratio', 'hours_from_own_baseline', 'is_weekend_num', 'iat_hours',
                                   'operational_friction_index', 'boundary_dissolution_index']].mean())

# ==========================================
# SAVE
# ==========================================
work.to_parquet('data/processed/github_clustered_SUPERSEDED.parquet', index=False)

trace3 = {
    'silhouette_k2': sil_scores[2],
    'silhouette_k3': sil_scores[3],
    'proporcion_cluster_anomalo_k2': cluster_sizes[minority_cluster],
    'chi2_fix': chi2_fix, 'p_fix': p_fix,
    'chi2_revert': chi2_revert, 'p_revert': p_revert,
    'sensitivity_seed_mean': np.mean(seed_results),
    'sensitivity_seed_std': np.std(seed_results),
}
pd.Series(trace3).to_csv('data/processed/fase3_trace_SUPERSEDED.csv', header=['valor'])
print("\nSaved (superseded output, not used downstream): "
      "github_clustered_SUPERSEDED.parquet and fase3_trace_SUPERSEDED.csv")
