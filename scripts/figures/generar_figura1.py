"""
Generates Figure 1: flow diagram of the complete methodological pipeline.
Output: figura1_pipeline.png (300 dpi, suitable for \\includegraphics in LaTeX)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

STAGES = [
    ("1. Stratified\nCollection", "7 GitHub repos\nmonthly sampling\n(2025-01 to 2026-07)"),
    ("2. Cleaning &\nTraceability", "bots, spurious\nidentities, churn\noutliers, explicit AI"),
    ("3. Feature\nEngineering", "operational friction,\nself-referential\nboundary dissolution"),
    ("4. Ground Truth\n(Isolation Forest)", "5-commit windows\n15% anomalous minority"),
    ("5. External\nValidation", "Defect Fix Rate /\nRevert Rate\n(Mann-Whitney U)"),
    ("6. Temporal\nStability", "stable trait vs.\nemergent state"),
    ("7. Oracle\nModeling", "LSTM vs. Random Forest\nGroupKFold (5 folds)"),
]

fig, ax = plt.subplots(figsize=(12, 3.2), dpi=300)
ax.set_xlim(0, len(STAGES))
ax.set_ylim(0, 1.6)
ax.axis("off")

box_w, box_h = 0.86, 1.05
y0 = 0.42

for i, (title, detail) in enumerate(STAGES):
    x0 = i + 0.07
    box = FancyBboxPatch(
        (x0, y0), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1, edgecolor="#2b2b2b", facecolor="#eef2f7"
    )
    ax.add_patch(box)
    ax.text(x0 + box_w / 2, y0 + box_h - 0.20, title,
            ha="center", va="top", fontsize=8.3, fontweight="bold")
    ax.text(x0 + box_w / 2, y0 + box_h - 0.48, detail,
            ha="center", va="top", fontsize=6.6, linespacing=1.35)

    if i < len(STAGES) - 1:
        arrow = FancyArrowPatch(
            (x0 + box_w + 0.01, y0 + box_h / 2),
            (x0 + 1.0, y0 + box_h / 2),
            arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#2b2b2b"
        )
        ax.add_patch(arrow)

plt.tight_layout()
plt.savefig("figura1_pipeline.png", dpi=300, bbox_inches="tight")
print("Saved: figura1_pipeline.png")
