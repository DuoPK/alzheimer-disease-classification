import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ConfusionMatrixPlotter:
    @staticmethod
    def plot(y_true, y_pred, labels=None, title="Confusion Matrix", figsize=(6, 5), cmap="Blues"):

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        plt.tight_layout()
        plt.show()
