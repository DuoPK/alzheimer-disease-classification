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
