from training.ClassificationMetrics import ClassificationMetrics

import numpy as np
from collections import defaultdict
import copy

class CrossValidator:
    def __init__(self, model, k=5, random_state=42, positive_label=1):
        """
        Parameters:
        - model: model
        - k: number of folds
        - random_state: for reproducibility
        - positive_label: class treated as positive in metrics
        """
        self.model = model
        self.k = k
        self.random_state = random_state
        self.positive_label = positive_label

    def split_stratified(self, X, y):
        np.random.seed(self.random_state)
        X = np.array(X)
        y = np.array(y)

        class_indices = defaultdict(list)
        for i, label in enumerate(y):
            class_indices[label].append(i)

        folds = [[] for _ in range(self.k)]
        for indices in class_indices.values():
            np.random.shuffle(indices)
            chunks = np.array_split(indices, self.k)
            for i in range(self.k):
                folds[i].extend(chunks[i])

        splits = []
        for i in range(self.k):
            test_idx = folds[i]
            train_idx = [idx for j in range(self.k) if j != i for idx in folds[j]]
            splits.append((train_idx, test_idx))
        return splits

    def evaluate(self, X, y):
        X = np.array(X)
        y = np.array(y)

        acc_scores = []
        f1_scores = []

        for train_idx, test_idx in self.split_stratified(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone the model for each fold (to reset weights/state)
            model = copy.deepcopy(self.model)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = ClassificationMetrics(y_test, y_pred, positive_label=self.positive_label)
            acc_scores.append(metrics.accuracy())
            f1_scores.append(metrics.f1_score())


        return {
            "mean_accuracy": np.mean(acc_scores),
            "mean_f1_score": np.mean(f1_scores),
            "accuracy_per_fold": acc_scores,
            "f1_per_fold": f1_scores
        }
