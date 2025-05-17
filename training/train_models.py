import os
import warnings

import pandas as pd
import time
import json
import logging
from datetime import datetime

from training.CrossValidator import CrossValidator
from training.models.RandomForestModel import RandomForestModel
from training.utils.ClassificationMetrics import ClassificationMetrics
from training.utils.StratifiedTrainTestSplitter import StratifiedTrainTestSplitter
from training.hyper_search.optuna_search import OptunaSearch
from training.utils.enums import DatasetType, ModelType
from training.utils.config import (
    RANDOM_STATE, SKLEARN_PARAMS, XGBOOST_PARAMS,
    CATBOOST_PARAMS, OPTUNA_PARAMS, set_random_state, CV, N_TRIALS, TEST_SIZE, MODEL_WITHOUT_N_JOBS_PARAM
)

# Import all models
from training.models.NeuralNetworkModel import NeuralNetworkModel
from training.models.SVCModel import SVCModel
from training.models.CatBoostModel import CatBoostModel
from training.models.XGBoostModel import XGBoostModel
from training.models.LogisticRegressionModel import LogisticRegressionModel
from training.models.GaussianNBModel import GaussianNBModel
from training.models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel
from training.models.VotingModel import VotingModel
from training.models.StackingModel import StackingModel
from scipy import linalg

warnings.filterwarnings("ignore", category=linalg.LinAlgWarning)

# Model mapping
MODEL_CLASSES = {
    # ModelType.NEURAL_NETWORK: NeuralNetworkModel,
    ModelType.SVM: SVCModel,
    ModelType.CATBOOST: CatBoostModel,
    ModelType.XGBOOST: XGBoostModel,
    ModelType.RANDOM_FOREST: RandomForestModel,
    ModelType.LOGISTIC_REGRESSION: LogisticRegressionModel,
    ModelType.GAUSSIAN_NB: GaussianNBModel,
    ModelType.QDA: QuadraticDiscriminantAnalysisModel,
    ModelType.VOTING: VotingModel,
    ModelType.STACKING: StackingModel
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
    """Get a model class from the model type"""
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
    
    # Split into train and test sets using custom StratifiedTrainTestSplitter
    splitter = StratifiedTrainTestSplitter(test_size=TEST_SIZE, random_state=RANDOM_STATE)
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
    
    # Initialize Optuna search with multiprocessing
    optuna_search = OptunaSearch(
        model_class=model_class,
        model_name=model_name,
        # input_size=X_train.shape[1] if model_type == ModelType.NEURAL_NETWORK else None,
        # output_size=len(np.unique(y)) if model_type == ModelType.NEURAL_NETWORK else None,
        n_trials=N_TRIALS,
        cv=CV,
        scoring='f1_score',
        **OPTUNA_PARAMS
    )
    
    # Run hyperparameter optimization
    optuna_search.fit(X_train, y_train, dataset_name)
    
    results["hyperparameter_search"] = {
        "best_params": optuna_search.best_params_,
        "best_score": optuna_search.best_score_,
        "trials": optuna_search.trials_
    }
    
    # Final training and testing with the best parameters
    logger.info("Starting final training and testing with best parameters")
    final_train_start = time.time()
    
    model_best_params = optuna_search.best_params_.copy()
    # Add appropriate multiprocessing parameters based on the model type
    if model_type == ModelType.XGBOOST:
        model_best_params.update(XGBOOST_PARAMS)
    elif model_type == ModelType.CATBOOST:
        model_best_params.update(CATBOOST_PARAMS)
    elif model_type == ModelType.SVM:
        model_best_params.update(MODEL_WITHOUT_N_JOBS_PARAM)
    elif model_type == ModelType.GAUSSIAN_NB or model_type == ModelType.QDA:
        pass
    # elif model_type == ModelType.NEURAL_NETWORK:
    #     model_best_params['input_size'] = X_train.shape[1]
    #     model_best_params['output_size'] = len(np.unique(y_train))
    else:
        model_best_params.update(SKLEARN_PARAMS)
    
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
    # Set a random state for all libraries
    set_random_state()
    
    # Train each model on each dataset
    for dataset_type in DatasetType.get_all_datasets():
        for model_type in ModelType.get_all_models():
            try:
                train_and_evaluate_model(model_type, dataset_type)
            except Exception as e:
                logging.error(f"Error training {model_type} on {dataset_type}: {str(e)}")

if __name__ == "__main__":
    main()
    