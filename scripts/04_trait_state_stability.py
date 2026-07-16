"""
Fase 4: Rasgo vs. Estado
Para cada autor con suficientes ventanas, se evalúa si sus ventanas anómalas (Isolation Forest)
están (a) persistentes y distribuidas a lo largo de todo su historial -> RASGO ESTABLE
o (b) concentradas en un episodio temporal acotado -> ESTADO EMERGENTE (alerta temprana defendible)
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy

MIN_WINDOWS_FOR_PROFILE = 10   # mínimo de ventanas por autor para perfilar temporalmente
win_df = pd.read_parquet('data/github_windows_groundtruth.parquet')
win_df = win_df.sort_values(['author_id', 'window_end_seq']).reset_index(drop=True)

author_counts = win_df.groupby('author_id').size()
eligible_authors = author_counts[author_counts >= MIN_WINDOWS_FOR_PROFILE].index
print(f"Autores con >= {MIN_WINDOWS_FOR_PROFILE} ventanas (elegibles para perfil temporal): {len(eligible_authors)} / {win_df['author_id'].nunique()}")

profiles = []
for author_id in eligible_authors:
    a = win_df[win_df['author_id'] == author_id].sort_values('window_end_seq').reset_index(drop=True)
    n = len(a)
    anomaly_rate = a['iso_anomaly'].mean()
    n_anomalous = a['iso_anomaly'].sum()

    # --- Concentración temporal: dividir el historial del autor en 4 cuartiles cronológicos ---
    a['quartile'] = pd.qcut(a.index, 4, labels=False, duplicates='drop')
    q_counts = a.groupby('quartile')['iso_anomaly'].sum()
    q_counts = q_counts.reindex(range(4), fill_value=0)

    if n_anomalous > 0:
        q_probs = q_counts / n_anomalous
        q_probs_nonzero = q_probs[q_probs > 0]
        ent = shannon_entropy(q_probs_nonzero, base=2)
        ent_norm = ent / np.log2(4)  # 0 = concentrado en 1 cuartil, 1 = perfectamente disperso
    else:
        ent_norm = np.nan

    # --- Autocorrelación lag-1 del risk_score continuo (persistencia de comportamiento) ---
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

# --- Clasificación rasgo vs. estado ---
# Solo clasificamos autores que tuvieron AL MENOS 1 ventana anómala (los que nunca tuvieron
# ninguna simplemente no aplican al esquema rasgo/estado, quedan como "sin señal")
has_signal = prof_df[prof_df['n_anomalous'] >= 2].copy()  # min 2 para que la entropía tenga sentido

rate_median = has_signal['anomaly_rate'].median()
entropy_median = has_signal['temporal_entropy_norm'].median()

def classify(row):
    high_rate = row['anomaly_rate'] >= rate_median
    high_entropy = row['temporal_entropy_norm'] >= entropy_median
    if high_rate and high_entropy:
        return 'rasgo_estable'       # persistente y disperso en el tiempo
    elif not high_rate and not high_entropy:
        return 'estado_emergente'    # poco frecuente y concentrado en un episodio
    else:
        return 'mixto'

has_signal['clasificacion'] = has_signal.apply(classify, axis=1)

print("\n=== Umbrales de clasificación (medianas) ===")
print(f"anomaly_rate mediana: {rate_median:.3f}")
print(f"temporal_entropy_norm mediana: {entropy_median:.3f}")

print("\n=== Distribución de clasificación (autores con >=2 ventanas anómalas) ===")
print(has_signal['clasificacion'].value_counts())
print(has_signal['clasificacion'].value_counts(normalize=True) * 100)

n_sin_senal = (prof_df['n_anomalous'] < 2).sum()
print(f"\nAutores sin señal suficiente (<2 ventanas anómalas): {n_sin_senal} / {len(prof_df)}")

print("\n=== Autocorrelación lag-1 del risk_score continuo (todos los autores elegibles) ===")
print(prof_df['autocorr_risk_lag1'].describe())
print(f"\n% de autores con autocorrelación positiva significativa (>0.2): "
      f"{(prof_df['autocorr_risk_lag1'] > 0.2).mean() * 100:.1f}%")

print("\n=== Perfil promedio por clasificación ===")
print(has_signal.groupby('clasificacion')[['n_windows', 'anomaly_rate', 'temporal_entropy_norm', 'autocorr_risk_lag1']].mean())

# ==========================================
# GUARDADO
# ==========================================
prof_df.to_parquet('data/fase4_author_profiles.parquet', index=False)
has_signal.to_csv('data/fase4_clasificacion_rasgo_estado.csv', index=False)

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
pd.Series(trace4).to_csv('data/fase4_trace.csv', header=['valor'])
print("\n✅ Guardado: fase4_author_profiles.parquet, fase4_clasificacion_rasgo_estado.csv, fase4_trace.csv")
