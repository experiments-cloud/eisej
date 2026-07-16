"""
Generates Figure 2: distribution of the three composite indices,
comparing the anomalous group (Isolation Forest) against the normal group.
Output: figura2_perfil_cluster.png (300 dpi)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

win_df = pd.read_parquet('data/github_windows_groundtruth.parquet')

FEATURES = {
    'operational_friction_index': 'Operational\nFriction Index',
    'boundary_dissolution_index': 'Boundary\nDissolution Index',
    'erratic_pacing_index_v2': 'Erratic Pacing\nIndex',
}

fig, axes = plt.subplots(1, 3, figsize=(10, 4), dpi=300, sharey=False)

for ax, (col, label) in zip(axes, FEATURES.items()):
    data_normal = win_df.loc[~win_df['iso_anomaly'], col].values
    data_anom = win_df.loc[win_df['iso_anomaly'], col].values

    bp = ax.boxplot(
        [data_normal, data_anom],
        labels=['Normal\n(n={:,})'.format(len(data_normal)), 'Anomalous\n(n={:,})'.format(len(data_anom))],
        patch_artist=True, showfliers=False, widths=0.55
    )
    colors = ['#a7c4e0', '#e08a8a']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor('#2b2b2b')
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('#2b2b2b')

    ax.set_title(label, fontsize=9.5, fontweight='bold')
    ax.axhline(0, color='#999999', linewidth=0.6, linestyle='--')
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Standardized value (z-score)', fontsize=9)

plt.tight_layout()
plt.savefig('figura2_perfil_cluster.png', dpi=300, bbox_inches='tight')
print("Saved: figura2_perfil_cluster.png")

# Supporting statistics for the figure caption / text
print("\nMedians by group:")
for col, label in FEATURES.items():
    med_n = win_df.loc[~win_df['iso_anomaly'], col].median()
    med_a = win_df.loc[win_df['iso_anomaly'], col].median()
    print(f"  {col}: normal={med_n:.3f}  anomalous={med_a:.3f}")
