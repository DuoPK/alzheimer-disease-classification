from sklearn.svm import SVC
from training.hyperparams_search.ModelOptimizer import ModelOptimizer

class SVCOptimizer(ModelOptimizer):
    def __init__(self, random_state=42, scoring='f1_macro', cv=5):
        self.random_state = random_state

        model = SVC(random_state=self.random_state)

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
