import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
from datetime import datetime
from training.utils.ClassificationMetrics import ClassificationMetrics


class ResultsLogger:
    def __init__(self):
        """Initialize ResultsLogger with directories for saving results"""
        self.results_dir = os.path.join('results', 'metrics')
        self.plots_dir = os.path.join('results', 'plots')
        self.metrics_file = os.path.join(self.results_dir, 'metrics.csv')

        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Load or create metrics DataFrame
        if os.path.exists(self.metrics_file):
            self.metrics_df = pd.read_csv(self.metrics_file)
        else:
            self.metrics_df = pd.DataFrame(columns=[
                'timestamp', 'dataset', 'model',
                'custom_cv_mean_accuracy', 'custom_cv_mean_f1_score',
                'custom_cv_std_accuracy', 'custom_cv_std_f1_score',
                'custom_cv_training_time',
                'sklearn_cv_mean_accuracy', 'sklearn_cv_mean_f1_score',
                'sklearn_cv_std_accuracy', 'sklearn_cv_std_f1_score',
                'sklearn_cv_training_time'
            ])

    def log_default_params_results(self, dataset_name, model_name, custom_cv_results, sklearn_cv_results):
        """Log results for default parameters"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        new_row = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'model': model_name,
            'custom_cv_mean_accuracy': custom_cv_results['mean_accuracy'],
            'custom_cv_mean_f1_score': custom_cv_results['mean_f1_score'],
            'custom_cv_std_accuracy': custom_cv_results['std_accuracy'],
            'custom_cv_std_f1_score': custom_cv_results['std_f1_score'],
            'custom_cv_training_time': custom_cv_results['training_time'],
            'sklearn_cv_mean_accuracy': sklearn_cv_results['mean_accuracy'],
            'sklearn_cv_mean_f1_score': sklearn_cv_results['mean_f1_score'],
            'sklearn_cv_std_accuracy': sklearn_cv_results['std_accuracy'],
            'sklearn_cv_std_f1_score': sklearn_cv_results['std_f1_score'],
            'sklearn_cv_training_time': sklearn_cv_results['training_time']
        }

        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([new_row])], ignore_index=True)
        self.metrics_df.to_csv(self.metrics_file, index=False)

    def save_confusion_matrix(self, y_true, y_pred, dataset_name, model_name):
        """Save confusion matrix plot"""
        # Calculate confusion matrix using ClassificationMetrics
        metrics = ClassificationMetrics(y_true, y_pred)
        cm = metrics.confusion_matrix()

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'confusion_matrix_{dataset_name}_{model_name}_{timestamp}.png'
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()

    def save_roc_curve(self, y_true, y_pred_proba, dataset_name, model_name):
        """Save ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} on {dataset_name}')
        plt.legend(loc="lower right")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'roc_curve_{dataset_name}_{model_name}_{timestamp}.png'
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()
