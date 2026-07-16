"""
Generates Figure 3: scatter plot of anomaly_rate x temporal_entropy_norm,
colored by classification (stable trait / emergent state / mixed),
with median lines marking the 4 quadrants.
Output: figura3_rasgo_estado.png (300 dpi)
"""
import pandas as pd
import matplotlib.pyplot as plt

has_signal = pd.read_csv('data/fase4_clasificacion_rasgo_estado.csv')

COLORS = {
    'rasgo_estable': '#c0392b',
    'estado_emergente': '#2980b9',
    'mixto': '#95a5a6',
}
LABELS_EN = {
    'rasgo_estable': 'Stable trait',
    'estado_emergente': 'Emergent state',
    'mixto': 'Mixed',
}

fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)

for cls, color in COLORS.items():
    sub = has_signal[has_signal['clasificacion'] == cls]
    ax.scatter(sub['anomaly_rate'], sub['temporal_entropy_norm'],
               s=32, alpha=0.75, color=color, edgecolor='white', linewidth=0.4,
               label=f"{LABELS_EN[cls]} (n={len(sub)})")

rate_med = has_signal['anomaly_rate'].median()
ent_med = has_signal['temporal_entropy_norm'].median()
ax.axvline(rate_med, color='#333333', linestyle='--', linewidth=0.8)
ax.axhline(ent_med, color='#333333', linestyle='--', linewidth=0.8)

ax.set_xlabel('Anomaly rate (per author)', fontsize=10)
ax.set_ylabel('Normalized temporal entropy', fontsize=10)
ax.legend(loc='lower right', fontsize=8, frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figura3_rasgo_estado.png', dpi=300, bbox_inches='tight')
print("Saved: figura3_rasgo_estado.png")
