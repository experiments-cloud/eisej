"""
Fase 3: Behavioral clustering (Ground Truth no supervisado)
- Selección de k (silhouette, k=2..6)
- Clustering final k=2 (framing teórico: perfil estable vs. anómalo)
- Validación externa: Defect Fix Rate / Revert Rate (chi-cuadrado), variables
  deliberadamente excluidas del feature space de clustering
- Sensitivity analysis: estabilidad de la proporción del cluster minoritario
  bajo distintas semillas y bajo k=3
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency

df = pd.read_parquet('data/github_features_final.parquet')

# --- Preparación de la matriz de features de comportamiento ---
# Excluimos deliberadamente is_fix / is_revert: se reservan para validación EXTERNA,
# no deben influir en la formación del cluster (evita circularidad trivial).
work = df.copy()
work['iat_hours'] = work.groupby('author_id')['iat_hours'].transform(lambda s: s.fillna(s.median()))
work['iat_hours'] = work['iat_hours'].fillna(work['iat_hours'].median())
work['log_iat'] = np.log1p(work['iat_hours'].clip(lower=0))
work['is_weekend_num'] = work['is_weekend'].astype(int)

# --- Composición de 2 índices teóricos (evita que un binario desbalanceado domine
#     la distancia euclidiana al mezclarse crudo con variables continuas) ---
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

print("=== Selección de k (silhouette score, k=2..6) ===")
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores[k] = sil
    print(f"k={k}: silhouette={sil:.4f}")

# --- Clustering final: k=2, framing teórico (estable vs. anómalo) ---
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
work['cluster'] = km2.fit_predict(X_scaled)

cluster_sizes = work['cluster'].value_counts(normalize=True).sort_index()
print("\n=== Distribución de clusters (k=2) ===")
print(cluster_sizes)

# Identificar cluster minoritario = "anómalo" (mayor churn destructivo + mayor desviación horaria)
cluster_profile = work.groupby('cluster')[['deletion_ratio', 'hours_from_own_baseline', 'is_weekend_num', 'iat_hours',
                                            'operational_friction_index', 'boundary_dissolution_index']].mean()
print("\n=== Perfil promedio por cluster ===")
print(cluster_profile)

minority_cluster = cluster_sizes.idxmin()
work['is_anomalous'] = work['cluster'] == minority_cluster
print(f"\nCluster anómalo identificado: {minority_cluster} ({cluster_sizes[minority_cluster]*100:.1f}% del corpus)")

# ==========================================
# VALIDACIÓN EXTERNA: Defect Fix Rate / Revert Rate
# ==========================================
print("\n=== Validación externa (variables NO usadas en el clustering) ===")
ext = work.groupby('is_anomalous')[['is_fix', 'is_revert']].mean()
print(ext)

# Chi-cuadrado: is_fix vs. cluster
ct_fix = pd.crosstab(work['is_anomalous'], work['is_fix'])
chi2_fix, p_fix, _, _ = chi2_contingency(ct_fix)
print(f"\nChi-cuadrado Defect Fix Rate: chi2={chi2_fix:.2f}, p={p_fix:.6f}")

ct_revert = pd.crosstab(work['is_anomalous'], work['is_revert'])
chi2_revert, p_revert, _, _ = chi2_contingency(ct_revert)
print(f"Chi-cuadrado Revert Rate: chi2={chi2_revert:.2f}, p={p_revert:.6f}")

# ==========================================
# SENSITIVITY ANALYSIS
# ==========================================
print("\n=== Sensitivity analysis: estabilidad de la proporción minoritaria ===")

print("\n-- Variación por semilla aleatoria (k=2) --")
seed_results = []
for seed in [0, 1, 7, 42, 123, 2024, 99999]:
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    prop = min(np.mean(labels == 0), np.mean(labels == 1))
    seed_results.append(prop)
    print(f"  seed={seed}: cluster minoritario = {prop*100:.2f}%")
print(f"  Media: {np.mean(seed_results)*100:.2f}%  |  Desv. estándar: {np.std(seed_results)*100:.2f}pp")

print("\n-- k=3 (subdivisión del cluster anómalo) --")
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels3 = km3.fit_predict(X_scaled)
work['cluster_k3'] = labels3
print(pd.Series(labels3).value_counts(normalize=True).sort_index())
print(work.groupby('cluster_k3')[['deletion_ratio', 'hours_from_own_baseline', 'is_weekend_num', 'iat_hours',
                                   'operational_friction_index', 'boundary_dissolution_index']].mean())

# ==========================================
# GUARDADO
# ==========================================
work.to_parquet('data/github_clustered.parquet', index=False)

trace3 = {
    'silhouette_k2': sil_scores[2],
    'silhouette_k3': sil_scores[3],
    'proporcion_cluster_anomalo_k2': cluster_sizes[minority_cluster],
    'chi2_fix': chi2_fix, 'p_fix': p_fix,
    'chi2_revert': chi2_revert, 'p_revert': p_revert,
    'sensitivity_seed_mean': np.mean(seed_results),
    'sensitivity_seed_std': np.std(seed_results),
}
pd.Series(trace3).to_csv('data/fase3_trace.csv', header=['valor'])
print("\n✅ Guardado: data/github_clustered.parquet y data/fase3_trace.csv")
