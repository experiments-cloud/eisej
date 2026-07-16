"""
Genera la Figura 6: tendencia trimestral de churn promedio y deletion ratio promedio,
2025Q1-2026Q3 (corpus definitivo, ya filtrado de señal explícita de IA).
Salida: figura6_tendencia_trimestral.png (300 dpi)
"""
import pandas as pd
import matplotlib.pyplot as plt

q = pd.read_csv('data/tendencia_trimestral.csv')
quarters = q['quarter'].astype(str)

fig, ax1 = plt.subplots(figsize=(7.2, 4.4), dpi=300)

color1 = '#c0392b'
ax1.set_xlabel('Quarter')
ax1.set_ylabel('Mean churn (lines)', color=color1, fontsize=10)
ax1.plot(quarters, q['churn_mean'], color=color1, marker='o', linewidth=1.8, label='Mean churn')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', rotation=30, labelsize=8)

ax2 = ax1.twinx()
color2 = '#2980b9'
ax2.set_ylabel('Mean deletion ratio', color=color2, fontsize=10)
ax2.plot(quarters, q['deletion_ratio_mean'], color=color2, marker='s', linewidth=1.8, linestyle='--', label='Mean deletion ratio')
ax2.tick_params(axis='y', labelcolor=color2)

# Marcar el último trimestre como incompleto (ventana de extracción corta a mitad de julio 2026)
ax1.axvspan(len(quarters) - 1.5, len(quarters) - 0.5, color='grey', alpha=0.12)
ax1.text(len(quarters) - 1, ax1.get_ylim()[1] * 0.97, 'partial\nquarter',
          ha='center', va='top', fontsize=7, style='italic', color='#555555')

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('figura6_tendencia_trimestral.png', dpi=300, bbox_inches='tight')
print("Guardado: figura6_tendencia_trimestral.png")
