import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import KBinsDiscretizer

file_path = "../ready_dataset_with_nulls_as_incorrect_data.csv"
df = pd.read_csv(file_path)

valid_ranges = {
    "Age": (1, 120),  # (60, 90)
    "BMI": (10, 100),  # (15, 40)
    "AlcoholConsumption": (0, 168),  # (0, 20), 1 unit = 10 ml pure ethanol, per week
    "PhysicalActivity": (0, 100),  # (0, 10), hours per week
    "DietQuality": (0, 10),  # Author's scale
    "SleepQuality": (0, 10),  # (4, 10), Author's scale
    "SystolicBP": (30.1, 260 - 0.1),  # (90, 180), Systolic blood pressure [mmHg]
    "DiastolicBP": (20.1, 220 - 0.1),  # (60, 120), Diastolic blood pressure [mmHg]
    "CholesterolTotal": (50.1, 600 - 0.1),  # (150, 300), Total cholesterol levels [mg/dL]
    "CholesterolLDL": (15.1, 450 - 0.1),  # (50, 200), Low-density lipoprotein cholesterol levels [mg/dL]
    "CholesterolHDL": (5.1, 200 - 0.1),  # (20, 100), High-density lipoprotein cholesterol levels [mg/dL]
    "CholesterolTriglycerides": (10.1, 1000 - 0.1),  # (50, 400), Triglycerides levels [mg/dL]
    "MMSE": (0, 30),  # Mini-Mental State Examination, standard scale
    "FunctionalAssessment": (0, 10),  # Author's scale
    "ADL": (0, 10),  # Author's scale, Activities of Daily Living score
}

binary_features = [
    "Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
    "Diabetes", "Depression", "HeadInjury", "Hypertension", "MemoryComplaints",
    "BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness"
]

categorical_features = {
    "EducationLevel": [0, 1, 2, 3],  # Valid levels: None, High School, Bachelor's, Higher
    "Ethnicity": [0, 1, 2, 3]  # {0: "Caucasian", 1: "African_American", 2: "Asian", 3: "Other"}
}

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


def print_changed_values(original_df, df):
    changed_cells = []
    for column in df.columns:
        null_mask = original_df[column].isnull()
        changed_indices = df.index[null_mask]
        for idx in changed_indices:
            diagnosis = df.at[idx, 'Diagnosis'] if 'Diagnosis' in df.columns else 'Brak diagnozy'
            changed_cells.append((idx, column, df.at[idx, column], diagnosis))

    print("Lista imputowanych wartości:")
    for idx, column, value, diagnosis in changed_cells:
        print(f"Indeks: {idx}, Cecha: {column}, Nowa wartość: {value}, Diagnoza: {diagnosis}")


df.to_csv("ready_dataset_with_imputed_values_3.csv", index=False)
# print_changed_values(original_df, df)
