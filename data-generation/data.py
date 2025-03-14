import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")

print(path)

df =  pd.read_csv(path+"/alzheimers_disease_data.csv")
df = df.iloc[:, 1:-1]

num_rows = df.shape[0]
sample_size = int(0.1 * num_rows)
random_rows = np.random.choice(df.index, size=sample_size, replace=False)

for row in random_rows:
    random_col = np.random.choice(df.columns)
    df.loc[row, random_col] = None

columns_map = {
    17: "SystolicBP",
    18: "DiastolicBP",
    19: "CholesterolTotal",
    20: "CholesterolLDL",
    21: "CholesterolHDL",
    22: "CholesterolTriglycerides"
}

extreme_ranges = {
    "SystolicBP": (0, 30, 260, 400),
    "DiastolicBP": (0, 20, 220, 400),
    "CholesterolTotal": (0, 50, 600, 1000),
    "CholesterolLDL": (0, 15, 450, 800),
    "CholesterolHDL": (0, 5, 200, 300),
    "CholesterolTriglycerides": (0, 10, 1000, 2000)
}

def losuj_ekstremalną_wartość(low1, high1, low2, high2):
    if np.random.rand() < 0.5:
        return np.random.randint(low1, high1 + 1)
    else:
        return np.random.randint(low2, high2 + 1)


num_rows = df.shape[0]
sample_size = int(0.02 * num_rows)
random_rows = np.random.choice(df.index, size=sample_size, replace=False)

for row in random_rows:
    for col_num, col_name in columns_map.items():
        if col_name in df.columns:
            low1, high1, low2, high2 = extreme_ranges[col_name]
            df.iloc[row, col_num] = losuj_ekstremalną_wartość(low1, high1, low2, high2)

output_file_path = "modified_extreme_dataset.csv"
df.to_csv(output_file_path, index=False)

print(f"Zmieniono wartości w {len(random_rows)} wierszach.")
print(f"Zapisano do: {output_file_path}")


df = pd.read_csv( "modified_extreme_dataset.csv")
sns.pairplot(df)
plt.show()
