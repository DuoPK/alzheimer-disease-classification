import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import KBinsDiscretizer

from fill_nan.utils import binary_features, categorical_features, valid_ranges, print_changed_values

file_path = "../ready_dataset_with_nulls_as_incorrect_data.csv"
df = pd.read_csv(file_path)

original_df = df.copy()

# dane numeryczne
num_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
# dane kategoryczne
cat_imputer = IterativeImputer(estimator=RandomForestClassifier(), max_iter=10, random_state=42)

# Imputacja cech binarnych
df[binary_features] = cat_imputer.fit_transform(df[binary_features])
# Imputacja cech kategorycznych
df[list(categorical_features.keys())] = cat_imputer.fit_transform(df[list(categorical_features.keys())])
# Imputacja cech ciągłych
df[list(valid_ranges.keys())] = num_imputer.fit_transform(df[list(valid_ranges.keys())])


df.to_csv("ready_dataset_with_imputed_values_3.csv", index=False)
# print_changed_values(original_df, df)
