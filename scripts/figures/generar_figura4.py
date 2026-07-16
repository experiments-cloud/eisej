"""
Generates Figure 4: boxplot comparing recall variability
across the 5 cross-validation folds, RF vs. LSTM.
Output: figura4_recall_variabilidad.png (300 dpi)
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/fase5_cv_resultados.csv')

fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=300)

data = [df['rf_recall'].values, df['lstm_recall'].values]
bp = ax.boxplot(data, tick_labels=['Random Forest', 'LSTM'],
                 patch_artist=True, widths=0.5)

colors = ['#a7c4e0', '#e08a8a']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_edgecolor('#2b2b2b')
for element in ['whiskers', 'caps', 'medians']:
    for line in bp[element]:
        line.set_color('#2b2b2b')

# Individual points per fold to show the real spread (n=5)
for i, col in enumerate(['rf_recall', 'lstm_recall'], start=1):
    x = [i] * len(df)
    ax.scatter(x, df[col], color='#2b2b2b', s=22, zorder=3, alpha=0.7)

ax.set_ylabel('Recall (5-fold cross-validation)', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('figura4_recall_variabilidad.png', dpi=300, bbox_inches='tight')
print("Saved: figura4_recall_variabilidad.png")

print("\nRF recall by fold:", df['rf_recall'].round(3).tolist())
print("LSTM recall by fold:", df['lstm_recall'].round(3).tolist())
