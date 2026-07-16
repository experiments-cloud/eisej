"""
Fase 3 (v2): Ground Truth no supervisado a nivel de VENTANA (no commit individual)
Unidad de análisis: ventanas deslizantes de 5 commits consecutivos por autor
(alineado con el diseño oracle SEQ_LEN=5 que se usará en Fase 5)
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu

SEQ_LEN = 5
df = pd.read_parquet('data/github_features_final.parquet')
df = df.sort_values(['author_id', 'commit_seq_number']).reset_index(drop=True)

# ==========================================
# CONSTRUCCIÓN DE VENTANAS DESLIZANTES (step=1) POR AUTOR
# ==========================================
windows = []
for author_id, grp in df.groupby('author_id'):
    grp = grp.reset_index(drop=True)
    n = len(grp)
    if n < SEQ_LEN:
        continue
    for start in range(0, n - SEQ_LEN + 1):
        w = grp.iloc[start:start + SEQ_LEN]
        windows.append({
            'author_id': author_id,
            'repo': w['repo'].iloc[-1],
            'window_end_date': w['dt'].iloc[-1],
            'window_end_seq': w['commit_seq_number'].iloc[-1],
            'deletion_ratio_mean': w['deletion_ratio'].mean(),
            'churn_mean': w['total_churn'].mean(),
            'boundary_dissolution_rate': w['boundary_dissolution_flag'].mean(),
            'hours_from_baseline_mean': w['hours_from_own_baseline'].mean(),
            'weekend_rate': w['is_weekend'].mean(),
            'iat_mean': w['iat_hours'].mean(),
            'iat_std': w['iat_hours'].std(),
            'is_fix_rate': w['is_fix'].mean(),
            'is_revert_rate': w['is_revert'].mean(),
        })

win_df = pd.DataFrame(windows)
win_df['iat_std'] = win_df['iat_std'].fillna(0)
print(f"Total ventanas construidas: {len(win_df)}")
print(f"Autores representados: {win_df['author_id'].nunique()}")
print(f"Ventanas promedio por autor: {len(win_df) / win_df['author_id'].nunique():.1f}")

# ==========================================
# FEATURE SPACE: 2 índices teóricos (mismo criterio que antes: evitar dominancia)
# ==========================================
z_del = StandardScaler().fit_transform(win_df[['deletion_ratio_mean']])[:, 0]
z_churn = StandardScaler().fit_transform(np.log1p(win_df[['churn_mean']]))[:, 0]
win_df['operational_friction_index'] = (z_del + z_churn) / 2

z_bd_rate = StandardScaler().fit_transform(win_df[['boundary_dissolution_rate']])[:, 0]
z_weekend = StandardScaler().fit_transform(win_df[['weekend_rate']])[:, 0]
win_df['boundary_dissolution_index'] = (z_bd_rate + z_weekend) / 2

win_df['log_iat_std'] = np.log1p(win_df['iat_std'])
z_iat_std = StandardScaler().fit_transform(win_df[['log_iat_std']])[:, 0]
win_df['erratic_pacing_index'] = z_iat_std

FEATURES = ['operational_friction_index', 'boundary_dissolution_index', 'erratic_pacing_index']
X = win_df[FEATURES].values
X_scaled = StandardScaler().fit_transform(X)

# ==========================================
# SELECCIÓN DE k
# ==========================================
print("\n=== Selección de k (silhouette, k=2..6) ===")
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores[k] = sil
    print(f"k={k}: silhouette={sil:.4f}")

# ==========================================
# CLUSTERING FINAL k=2
# ==========================================
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
win_df['cluster'] = km2.fit_predict(X_scaled)
cluster_sizes = win_df['cluster'].value_counts(normalize=True).sort_index()
print("\n=== Distribución de clusters (k=2, nivel ventana) ===")
print(cluster_sizes)

profile_cols = ['deletion_ratio_mean', 'churn_mean', 'boundary_dissolution_rate',
                 'weekend_rate', 'iat_mean', 'iat_std',
                 'operational_friction_index', 'boundary_dissolution_index', 'erratic_pacing_index']
print("\n=== Perfil promedio por cluster ===")
print(win_df.groupby('cluster')[profile_cols].mean().T)

minority_cluster = cluster_sizes.idxmin()
win_df['is_anomalous'] = win_df['cluster'] == minority_cluster
print(f"\nCluster anómalo: {minority_cluster} ({cluster_sizes[minority_cluster]*100:.1f}% de las ventanas)")

# ==========================================
# VALIDACIÓN EXTERNA (Mann-Whitney U, variables no usadas en clustering)
# ==========================================
print("\n=== Validación externa ===")
grp_anom = win_df[win_df['is_anomalous']]
grp_healthy = win_df[~win_df['is_anomalous']]

u_fix, p_fix = mannwhitneyu(grp_anom['is_fix_rate'], grp_healthy['is_fix_rate'], alternative='two-sided')
u_revert, p_revert = mannwhitneyu(grp_anom['is_revert_rate'], grp_healthy['is_revert_rate'], alternative='two-sided')

print(f"Defect Fix Rate  -> anómalo: {grp_anom['is_fix_rate'].mean():.4f}  |  sano: {grp_healthy['is_fix_rate'].mean():.4f}  |  Mann-Whitney p={p_fix:.6f}")
print(f"Revert Rate      -> anómalo: {grp_anom['is_revert_rate'].mean():.4f}  |  sano: {grp_healthy['is_revert_rate'].mean():.4f}  |  Mann-Whitney p={p_revert:.6f}")

# ==========================================
# SENSITIVITY ANALYSIS
# ==========================================
print("\n=== Sensitivity analysis (semillas) ===")
seed_results = []
for seed in [0, 1, 7, 42, 123, 2024, 99999]:
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    prop = min(np.mean(labels == 0), np.mean(labels == 1))
    seed_results.append(prop)
    print(f"  seed={seed}: minoritario = {prop*100:.2f}%")
print(f"  Media: {np.mean(seed_results)*100:.2f}%  |  Desv. estándar: {np.std(seed_results)*100:.2f}pp")

# ==========================================
# GUARDADO
# ==========================================
win_df.to_parquet('data/github_windows_clustered.parquet', index=False)
trace3 = {
    'total_ventanas': len(win_df),
    'autores_con_ventanas': win_df['author_id'].nunique(),
    'silhouette_k2': sil_scores[2],
    'mejor_k_por_silhouette': max(sil_scores, key=sil_scores.get),
    'proporcion_cluster_anomalo_k2': cluster_sizes[minority_cluster],
    'fix_rate_anomalo': grp_anom['is_fix_rate'].mean(),
    'fix_rate_sano': grp_healthy['is_fix_rate'].mean(),
    'p_fix_mannwhitney': p_fix,
    'revert_rate_anomalo': grp_anom['is_revert_rate'].mean(),
    'revert_rate_sano': grp_healthy['is_revert_rate'].mean(),
    'p_revert_mannwhitney': p_revert,
    'sensitivity_seed_mean': np.mean(seed_results),
    'sensitivity_seed_std': np.std(seed_results),
}
pd.Series(trace3).to_csv('data/fase3_trace_windows.csv', header=['valor'])
print("\n✅ Guardado: data/github_windows_clustered.parquet y data/fase3_trace_windows.csv")
