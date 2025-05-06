import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import time
import json
import logging
from datetime import datetime

from training.CrossValidator import CrossValidator
from training.utils.ClassificationMetrics import ClassificationMetrics
from training.utils.StratifiedTrainTestSplitter import StratifiedTrainTestSplitter
from training.hyper_search.optuna_search import OptunaSearch
from training.utils.enums import DatasetType, ModelType

# Import all models
from training.models.NeuralNetworkModel import NeuralNetworkModel
from training.models.SVCModel import SVCModel
from training.models.CatBoostModel import CatBoostModel
from training.models.XGBoostModel import XGBoostModel
from training.models.DecisionTreeModel import DecisionTreeModel

# Model mapping
MODEL_CLASSES = {
    ModelType.NEURAL_NETWORK: NeuralNetworkModel,
    ModelType.SVM: SVCModel,
    ModelType.CATBOOST: CatBoostModel,
    ModelType.XGBOOST: XGBoostModel,
    ModelType.DECISION_TREE: DecisionTreeModel
}

# Setup logging
def setup_logging(dataset_name, model_name):
    log_dir = "training/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{dataset_name}_{model_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('Diagnosis', axis=1).values
    y = df['Diagnosis'].values
    return X, y

def get_model_class(model_type: ModelType):
    """Get model class from the model type"""
    return MODEL_CLASSES[model_type]

def train_and_evaluate_model(model_type: ModelType, dataset_type: DatasetType):
    # Setup logging
    dataset_name = DatasetType.get_dataset_name(dataset_type)
    model_name = ModelType.get_model_class_name(model_type)
    logger = setup_logging(dataset_name, model_name)
    
    logger.info(f"Starting training for {model_name} on {dataset_name}")
    
    # Load data
    dataset_path = DatasetType.get_dataset_path(dataset_type)
    X, y = load_dataset(dataset_path)
    
    # Split into train and test sets using StratifiedTrainTestSplitter
    splitter = StratifiedTrainTestSplitter(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    # Initialize results dictionary
    results = {
        "dataset": dataset_name,
        "model": model_name,
        "hyperparameter_search": {},
        "custom_cv_results": {},
        "sklearn_cv_results": {},
        "final_test_results": {}
    }
    
    logger.info("Starting hyperparameter search with Optuna")
    model_class = get_model_class(model_type)
    
    # Initialize Optuna search
    optuna_search = OptunaSearch(
        model_class=model_class,
        model_name=model_name,
        input_size=X_train.shape[1] if model_type == ModelType.NEURAL_NETWORK else None,
        output_size=len(np.unique(y)) if model_type == ModelType.NEURAL_NETWORK else None,
        n_trials=2,
        cv=5,
        scoring='f1_score'
    )
    
    # Run hyperparameter optimization
    optuna_search.fit(X_train, y_train)
    
    results["hyperparameter_search"] = {
        "best_params": optuna_search.best_params_,
        "best_score": optuna_search.best_score_,
        "trials": optuna_search.trials_
    }
    
    # Custom Cross-Validation with best parameters
    logger.info("Starting custom cross-validation with best parameters")
    custom_cv_start = time.time()
    
    model_best_params = optuna_search.best_params_.copy()
    # Ensure input_size and output_size are included for neural network
    if model_type == ModelType.NEURAL_NETWORK:
        model_best_params['input_size'] = X_train.shape[1]
        model_best_params['output_size'] = len(np.unique(y_train))
    
    base_model = model_class(**model_best_params)
    custom_cv = CrossValidator(base_model, k=5, random_state=42)
    custom_cv_results = custom_cv.evaluate(X_train, y_train)
    
    custom_cv_time = time.time() - custom_cv_start
    results["custom_cv_results"] = {
        "metrics": custom_cv_results,
        "training_time": custom_cv_time
    }
    
    # Sklearn Cross-Validation with best parameters
    logger.info("Starting sklearn cross-validation with best parameters")
    sklearn_cv_start = time.time()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sk_acc_scores = []
    sk_f1_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model = model_class(**model_best_params)
        if hasattr(model, 'train'):
            model.train(X_fold_train, y_fold_train)
        
        y_pred = model.predict(X_fold_val)
        sk_acc_scores.append(accuracy_score(y_fold_val, y_pred))
        sk_f1_scores.append(f1_score(y_fold_val, y_pred))
    
    sklearn_cv_time = time.time() - sklearn_cv_start
    results["sklearn_cv_results"] = {
        "mean_accuracy": np.mean(sk_acc_scores),
        "std_accuracy": np.std(sk_acc_scores),
        "mean_f1_score": np.mean(sk_f1_scores),
        "std_f1_score": np.std(sk_f1_scores),
        "training_time": sklearn_cv_time
    }
    
    # Final training and testing with best parameters
    logger.info("Starting final training and testing with best parameters")
    final_train_start = time.time()
    
    final_model = model_class(**model_best_params)
    if hasattr(final_model, 'train'):
        final_model.train(X_train, y_train)
    
    y_pred = final_model.predict(X_test)
    metrics = ClassificationMetrics(y_test, y_pred)
    
    final_train_time = time.time() - final_train_start
    results["final_test_results"] = {
        "metrics": metrics.summary(),
        "training_time": final_train_time
    }
    
    # Save results
    results_dir = "training/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{dataset_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Training completed. Results saved to {results_file}")
    return results

def main():
    # Train each model on each dataset
    for dataset_type in DatasetType.get_all_datasets():
        for model_type in ModelType.get_all_models():
            try:
                train_and_evaluate_model(model_type, dataset_type)
            except Exception as e:
                logging.error(f"Error training {model_type} on {dataset_type}: {str(e)}")

if __name__ == "__main__":
    main()
    