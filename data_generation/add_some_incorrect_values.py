import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")
print(path)

df = pd.read_csv(path + "/alzheimers_disease_data.csv")
df = df.iloc[:, 1:-1]

num_rows = df.shape[0]
sample_size = int(0.1 * num_rows)
random_rows = np.random.choice(df.index, size=sample_size, replace=False)

for row in random_rows:
    random_col = np.random.choice(df.columns)
    df.loc[row, random_col] = None

extreme_ranges = {
    "SystolicBP": (0, 30, 260, 400),
    "DiastolicBP": (0, 20, 220, 400),
    "CholesterolTotal": (0, 50, 600, 1000),
    "CholesterolLDL": (0, 15, 450, 800),
    "CholesterolHDL": (0, 5, 200, 300),
    "CholesterolTriglycerides": (0, 10, 1000, 2000)
}


def get_random_extreme_value(low1, high1, low2, high2):
    if np.random.rand() < 0.5:
        return np.random.randint(low1, high1 + 1)
    else:
        return np.random.randint(low2, high2 + 1)


target_columns = [col for col in df.columns if col in extreme_ranges]
num_rows = df.shape[0]
sample_size = int(0.02 * num_rows)
random_rows = np.random.choice(df.index, size=sample_size, replace=False)

for row in random_rows:
    for col_name in target_columns:
        low1, high1, low2, high2 = extreme_ranges[col_name]
        df.loc[row, col_name] = get_random_extreme_value(low1, high1, low2, high2)

output_file_path = "modified_extreme_dataset.csv"
df.to_csv(output_file_path, index=False)

