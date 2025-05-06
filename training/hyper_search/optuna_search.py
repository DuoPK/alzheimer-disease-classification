import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
import time
import logging

class OptunaSearch:
    def __init__(self, model_class, model_name, input_size=None, output_size=None, 
                 n_trials=50, cv=5, scoring='f1_score', random_state=42):
        """
        Parameters:
        - model_class: class of the model to optimize
        - model_name: name of the model (used to get parameter space)
        - input_size: input size for neural network
        - output_size: output size for neural network
        - n_trials: number of optimization trials
        - cv: number of cross-validation folds
        - scoring: metric to optimize ('accuracy' or 'f1_score')
        - random_state: random seed for reproducibility
        """
        self.model_class = model_class
        self.model_name = model_name
        self.input_size = input_size
        self.output_size = output_size
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.trials_ = []
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False
        
    def _objective(self, trial, X, y):
        """Objective function for Optuna optimization"""
        # Get parameter suggestions based on model type
        if self.model_name == 'NeuralNetworkModel':
            params = self._suggest_neural_network_params(trial)
            # Log neural network structure for each trial
            model = self.model_class(**params)
            self.logger.info(f"\nTrial {len(self.trials_) + 1} - {model.get_model_structure()}")
        elif self.model_name == 'SVCModel':
            params = self._suggest_svm_params(trial)
        elif self.model_name == 'CatBoostModel':
            params = self._suggest_catboost_params(trial)
        elif self.model_name == 'XGBoostModel':
            params = self._suggest_xgboost_params(trial)
        elif self.model_name == 'DecisionTreeModel':
            params = self._suggest_decision_tree_params(trial)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Initialize and train model
            model = self.model_class(**params)
            if hasattr(model, 'train'):
                model.train(X_train, y_train)
            
            y_pred = model.predict(X_val)
            
            # Calculate score
            if self.scoring == 'accuracy':
                score = np.mean(y_pred == y_val)
            elif self.scoring == 'f1_score':
                from sklearn.metrics import f1_score
                score = f1_score(y_val, y_pred)
            else:
                raise ValueError(f"Unknown scoring metric: {self.scoring}")
            
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Store trial results
        self.trials_.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score
        })
        
        return mean_score
    
    def _suggest_neural_network_params(self, trial):
        # First suggest the number of layers
        n_layers = trial.suggest_int('n_layers', 1, 3)
        
        # Then suggest the size for each layer
        hidden_sizes = []
        for i in range(n_layers):
            size = trial.suggest_int(f'layer_{i}_size', 32, 256, step=32)
            hidden_sizes.append(size)
        
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_sizes': hidden_sizes,
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'epochs': trial.suggest_int('epochs', 50, 200, step=50),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.1)
        }
    
    def _suggest_svm_params(self, trial):
        return {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 0.1, 0.01])
        }
    
    def _suggest_catboost_params(self, trial):
        return {
            'iterations': trial.suggest_int('iterations', 100, 500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8, step=2),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 5, step=2)
        }
    
    def _suggest_xgboost_params(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7, step=2),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5, step=2)
        }
    
    def _suggest_decision_tree_params(self, trial):
        return {
            'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 15]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=4),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, step=1)
        }
    
    def fit(self, X, y):
        """Find the best parameters using Optuna optimization"""
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X, y), 
                      n_trials=self.n_trials)
        
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        # Log results
        self.logger.info(f"Best parameters: {self.best_params_}")
        self.logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self 