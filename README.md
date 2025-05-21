# Alzheimer's Disease classification

## Datasets
In root folder:
- Original: `alzheimers_disease_data.csv`
- After add nulls and extreme values: `modified_extreme_dataset.csv `
  <br>**We perform analysis on this data (histograms and boxplots).**
- After extreme values change to nulls: `ready_dataset_to_preprocessing.csv`
  <br>a) **We perform analysis again on this data:**
  - Scatterplots, pairplots, heatmap.
  - Analysis of individual data (e.g. what part of sick patients has a higher value of the `P` feature than the `x` value).

### Add null / incorrect values
`cd data_generation`<br>
`py add_some_incorrect_values.py`

### Change extreme values to nulls
And remove rows with missing dependent variable (diagnosis).<br>
`cd analyze`<br>
`py main.py`

## Preprocessing
### Fill missing values
- Mean and mode,
- Median and mode,
- IterativeImputer (BayesianRidge) and RandomForestClassifier.<br>

### Features scaling
- min-max scaling,
- standardization.

`cd data_generation`<br>
`py generator.py`

Datasets ready for classification are in the `data_generation/data` folder.

## Training

### Run
`cd training`<br>
`python train_models.py`

### Results
#### 1. Results of all models trained on all datasets with default parameters
- `training/results/metrics/metrics.csv`<br>
`custom_cv_[metric]` columns contain metrics calculated using custom cross-validation.
`sklearn_cv_[metric]` columns contain metrics calculated using sklearn cross-validation.
- `training/results/plots/`<br>
Contains auc and confusion matrix plots.

#### 2. Results of all models trained on all datasets with the best parameters
Parameters are searched using Optuna.
- `training/training/results/`<br>
JSON files contain data from each trial.
```JSON
{
  "hyperparameter_search": {
    "best_params": {
      # Best parameters for model
    },
    "best_score": {
      # Best score (f1-score) for model in cross-validation (mean of all folds)
    },
  },
  "final_test_results": {
    "metrics": {
      # Metrics for model on test set
    },
    "training_time": {
      # Training time on train set
    },
  }
}
```

#### 3. Optuna results
File: `training/optuna_results.db`<br>
Run: `optuna-dashboard sqlite:///alzheimer_classification.db`

