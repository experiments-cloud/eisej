"""
Genera la Figura 5: matrices de confusión agregadas (out-of-fold) para RF y LSTM.
Salida: figura5_matrices_confusion.png (300 dpi)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

d = np.load('data/fase5_oof_predictions.npz')
y_true, oof_rf, oof_lstm = d['y_true'], d['oof_rf'], d['oof_lstm']

cm_rf = confusion_matrix(y_true, oof_rf)
cm_lstm = confusion_matrix(y_true, oof_lstm)

fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=300)

for ax, cm, title in zip(axes, [cm_rf, cm_lstm], ['Random Forest', 'LSTM']):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomalous'], fontsize=9)
    ax.set_yticklabels(['Normal', 'Anomalous'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center',
                     fontsize=11, fontweight='bold',
                     color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig('figura5_matrices_confusion.png', dpi=300, bbox_inches='tight')
print("Guardado: figura5_matrices_confusion.png")
print("\nRF confusion matrix:\n", cm_rf)
print("\nLSTM confusion matrix:\n", cm_lstm)
