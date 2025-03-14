import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from fill_nan.utils import binary_features, categorical_features, valid_ranges, print_changed_values

file_path = "../ready_dataset_with_nulls_as_incorrect_data.csv"
df = pd.read_csv(file_path)

original_df = df.copy()


# Imputacja cech ciągłych dla każdej grupy Diagnosis
for feature, (lower, upper) in valid_ranges.items():
    for diagnosis in [0, 1]:
        group = df[df["Diagnosis"] == diagnosis]
        known = group.dropna(subset=[feature])
        unknown = group[group[feature].isnull()]

        if not unknown.empty:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(known[['Diagnosis']], known[feature])  # Tylko 'Diagnosis' jako cecha predykcyjna
            predictions = model.predict(unknown[['Diagnosis']])  # Predykcja brakujących wartości
            df.loc[df['Diagnosis'] == diagnosis, feature] = df.loc[df['Diagnosis'] == diagnosis, feature].fillna(pd.Series(predictions, index=unknown.index))

# Zaokrąglenie do odpowiednich zakresów
for feature, (lower, upper) in valid_ranges.items():
    df[feature] = df[feature].clip(lower, upper)

# Imputacja cech binarnych
for feature in binary_features:
    for diagnosis in [0, 1]:
        group = df[df["Diagnosis"] == diagnosis]
        known = group.dropna(subset=[feature])
        unknown = group[group[feature].isnull()]

        if not unknown.empty:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(known[['Diagnosis']], known[feature])
            predictions = model.predict(unknown[['Diagnosis']])  # Predykcja brakujących wartości
            df.loc[df['Diagnosis'] == diagnosis, feature] = df.loc[df['Diagnosis'] == diagnosis, feature].fillna(pd.Series(predictions, index=unknown.index))

# Imputacja cech kategorycznych
for feature, valid_values in categorical_features.items():
    for diagnosis in [0, 1]:
        group = df[df["Diagnosis"] == diagnosis]
        known = group.dropna(subset=[feature])
        unknown = group[group[feature].isnull()]

        if not unknown.empty:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(known[['Diagnosis']], known[feature])
            predictions = model.predict(unknown[['Diagnosis']])  # Predykcja brakujących wartości
            df.loc[df['Diagnosis'] == diagnosis, feature] = df.loc[df['Diagnosis'] == diagnosis, feature].fillna(pd.Series(predictions, index=unknown.index))

df.to_csv("ready_dataset_with_imputed_values_3-v2.csv", index=False)
# print_changed_values(original_df, df)
