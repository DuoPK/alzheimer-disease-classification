from enum import Enum, auto
from typing import List

class DatasetType(Enum):
    """Enum for available datasets"""
    MEAN_STANDARD = "mean_standard.csv"
    MEDIAN_STANDARD = "median_standard.csv"
    IMPUTE_STANDARD = "impute_standard.csv"
    MEAN_MINMAX = "mean_minmax.csv"
    MEDIAN_MINMAX = "median_minmax.csv"
    IMPUTE_MINMAX = "impute_minmax.csv"
    
    @classmethod
    def get_all_datasets(cls) -> List['DatasetType']:
        """Get all available datasets"""
        return list(cls)
    
    @classmethod
    def get_dataset_path(cls, dataset_type: 'DatasetType') -> str:
        """Get full path to the dataset"""
        return f"data_generation/data/{dataset_type.value}"
    
    @classmethod
    def get_dataset_name(cls, dataset_type: 'DatasetType') -> str:
        """Get dataset name without extension"""
        return dataset_type.value.replace('.csv', '')

class ModelType(Enum):
    """Enum for available models"""
    NEURAL_NETWORK = "NeuralNetworkModel"
    SVM = "SVCModel"
    CATBOOST = "CatBoostModel"
    XGBOOST = "XGBoostModel"
    DECISION_TREE = "DecisionTreeModel"
    
    @classmethod
    def get_all_models(cls) -> List['ModelType']:
        """Get all available models"""
        return list(cls)
    
    @classmethod
    def get_model_class_name(cls, model_type: 'ModelType') -> str:
        """Get model class name"""
        return model_type.value 