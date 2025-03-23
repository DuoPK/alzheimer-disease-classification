import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge

from features import binary_features, categorical_features, range_features, int_features
from data_generation.utils import print_changed_values

file_path = "../ready_dataset_to_preprocessing.csv"
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
df[list(range_features.keys())] = num_imputer.fit_transform(df[list(range_features.keys())])
# Zaokrąglenie do liczb całkowitych
df[int_features] = df[int_features].applymap(lambda x: round(x))

df.to_csv("filled_with_imputed_values.csv", index=False)
print_changed_values(original_df, df)
