"""
Fase 3 (v3): Comparación de dos enfoques de Ground Truth
  A) Índice continuo de riesgo + umbral por percentil (sin pretender "descubrimiento" no supervisado)
  B) Detección de anomalías diseñada para hallar minorías: Isolation Forest y Gaussian Mixture Model
Se reporta cuál es más robusto (separación real en variables de validación externa,
consistencia entre métodos, estabilidad ante semillas).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from scipy.stats import mannwhitneyu

win_df = pd.read_parquet('data/github_windows_clustered.parquet')

# --- Corrección: erratic_pacing debe capturar IRREGULARIDAD relativa al propio ritmo,
#     no solo magnitud de IAT (que en realidad refleja frecuencia de commit, un rasgo
#     de rol/dedicación, no necesariamente riesgo). Se usa coeficiente de variación. ---
win_df['iat_cv'] = win_df['iat_std'] / win_df['iat_mean'].replace(0, np.nan)
win_df['iat_cv'] = win_df['iat_cv'].fillna(win_df['iat_cv'].median())
z_cv = StandardScaler().fit_transform(np.log1p(win_df[['iat_cv']]))[:, 0]
win_df['erratic_pacing_index_v2'] = z_cv

FEATURES = ['operational_friction_index', 'boundary_dissolution_index', 'erratic_pacing_index_v2']
X = win_df[FEATURES].values
X_scaled = StandardScaler().fit_transform(X)

# ==========================================
# ENFOQUE A: ÍNDICE CONTINUO DE RIESGO (percentiles)
# ==========================================
win_df['risk_score'] = X_scaled.mean(axis=1)

results = {}
for pct in [0.10, 0.15, 0.20]:
    threshold = win_df['risk_score'].quantile(1 - pct)
    win_df[f'high_risk_p{int(pct*100)}'] = win_df['risk_score'] >= threshold

print("=== ENFOQUE A: Índice continuo + umbral por percentil ===")
for pct in [10, 15, 20]:
    col = f'high_risk_p{pct}'
    grp_hi = win_df[win_df[col]]
    grp_lo = win_df[~win_df[col]]
    u, p = mannwhitneyu(grp_hi['is_fix_rate'], grp_lo['is_fix_rate'], alternative='two-sided')
    print(f"  p{pct}: n={grp_hi.shape[0]} ({grp_hi.shape[0]/len(win_df)*100:.1f}%) | "
          f"fix_rate alto={grp_hi['is_fix_rate'].mean():.3f} vs bajo={grp_lo['is_fix_rate'].mean():.3f} | p={p:.2e}")
    results[f'A_p{pct}_n'] = grp_hi.shape[0]
    results[f'A_p{pct}_fixrate_hi'] = grp_hi['is_fix_rate'].mean()
    results[f'A_p{pct}_fixrate_lo'] = grp_lo['is_fix_rate'].mean()
    results[f'A_p{pct}_pvalue'] = p

# ==========================================
# ENFOQUE B1: ISOLATION FOREST
# ==========================================
print("\n=== ENFOQUE B1: Isolation Forest (contamination=0.15) ===")
iso = IsolationForest(contamination=0.15, random_state=42, n_estimators=200)
win_df['iso_anomaly'] = iso.fit_predict(X_scaled) == -1
n_iso = win_df['iso_anomaly'].sum()
print(f"  Anómalos detectados: {n_iso} ({n_iso/len(win_df)*100:.1f}%)")

grp_iso_hi = win_df[win_df['iso_anomaly']]
grp_iso_lo = win_df[~win_df['iso_anomaly']]
u_iso, p_iso = mannwhitneyu(grp_iso_hi['is_fix_rate'], grp_iso_lo['is_fix_rate'], alternative='two-sided')
print(f"  fix_rate anómalo={grp_iso_hi['is_fix_rate'].mean():.3f} vs normal={grp_iso_lo['is_fix_rate'].mean():.3f} | p={p_iso:.2e}")

# Sensitivity: estabilidad de Isolation Forest ante semillas
iso_seed_props = []
for seed in [0, 1, 7, 42, 123]:
    iso_s = IsolationForest(contamination=0.15, random_state=seed, n_estimators=200)
    labels = iso_s.fit_predict(X_scaled) == -1
    iso_seed_props.append(labels.mean())
print(f"  Estabilidad entre semillas: media={np.mean(iso_seed_props)*100:.2f}%, std={np.std(iso_seed_props)*100:.2f}pp")

# ==========================================
# ENFOQUE B2: GAUSSIAN MIXTURE MODEL (covarianza libre -> permite clusters de distinto tamaño)
# ==========================================
print("\n=== ENFOQUE B2: Gaussian Mixture Model (k=2, covariance_type='full') ===")
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=10)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_sizes = pd.Series(gmm_labels).value_counts(normalize=True).sort_index()
print(f"  Proporciones: {dict(gmm_sizes)}")
minority_gmm = gmm_sizes.idxmin()
win_df['gmm_anomaly'] = gmm_labels == minority_gmm
n_gmm = win_df['gmm_anomaly'].sum()
print(f"  Cluster minoritario: {n_gmm} ({n_gmm/len(win_df)*100:.1f}%)")

grp_gmm_hi = win_df[win_df['gmm_anomaly']]
grp_gmm_lo = win_df[~win_df['gmm_anomaly']]
u_gmm, p_gmm = mannwhitneyu(grp_gmm_hi['is_fix_rate'], grp_gmm_lo['is_fix_rate'], alternative='two-sided')
print(f"  fix_rate minoritario={grp_gmm_hi['is_fix_rate'].mean():.3f} vs mayoritario={grp_gmm_lo['is_fix_rate'].mean():.3f} | p={p_gmm:.2e}")

# ==========================================
# CONVERGENCIA ENTRE MÉTODOS (validez convergente)
# ==========================================
print("\n=== Convergencia entre métodos (¿marcan a los mismos autores/ventanas?) ===")
def jaccard(a, b):
    a, b = set(a[a].index), set(b[b].index)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)

j_A15_iso = jaccard(win_df['high_risk_p15'], win_df['iso_anomaly'])
j_A15_gmm = jaccard(win_df['high_risk_p15'], win_df['gmm_anomaly'])
j_iso_gmm = jaccard(win_df['iso_anomaly'], win_df['gmm_anomaly'])
print(f"  Jaccard (percentil15 vs IsolationForest): {j_A15_iso:.3f}")
print(f"  Jaccard (percentil15 vs GMM): {j_A15_gmm:.3f}")
print(f"  Jaccard (IsolationForest vs GMM): {j_iso_gmm:.3f}")

# ==========================================
# TABLA COMPARATIVA FINAL
# ==========================================
print("\n=== TABLA COMPARATIVA DE ROBUSTEZ ===")
comparison = pd.DataFrame({
    'metodo': ['A: percentil 15%', 'B1: Isolation Forest', 'B2: GMM (k=2, full cov)'],
    'proporcion_pct': [
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
    'estabilidad_semillas_std_pp': [np.nan, np.std(iso_seed_props)*100, np.nan],
})
print(comparison.to_string(index=False))

# ==========================================
# GUARDADO
# ==========================================
win_df.to_parquet('data/github_windows_groundtruth.parquet', index=False)
comparison.to_csv('data/fase3_comparacion_metodos.csv', index=False)
pd.Series(results).to_csv('data/fase3_trace_enfoqueA.csv', header=['valor'])
print("\n✅ Guardado: github_windows_groundtruth.parquet, fase3_comparacion_metodos.csv, fase3_trace_enfoqueA.csv")
