import pandas as pd

from data_generation.csv_generator import CSV
from data_generation.scaller import DataScaler


class DataFiller:
    int_columns = ['age', 'gender', 'ethnicity', 'education_level', 'smoking']

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.numeric_cols = self.df.select_dtypes(include='number').columns

    def fill_na_with_mean(self):
        df_filled = self.df.copy()
        for col in self.numeric_cols:
            mean_value = df_filled[col].mean()
            if col in DataFiller.int_columns:
                mean_value = round(mean_value)
            else:
                mean_value = round(mean_value, 4)
            df_filled[col] =df_filled[col].fillna(mean_value)
        return df_filled

    def fill_na_with_median(self):
        df_filled = self.df.copy()
        for col in self.numeric_cols:
            median_value = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(median_value)
        return df_filled



if __name__ == "__main__":
    cleaner = DataFiller("../ready_dataset_with_nulls_as_incorrect_data.csv")
    dataScaler = DataScaler()

    df_mean_filled = cleaner.fill_na_with_mean()
    CSV.save(df_mean_filled, "filled_with_mean.csv")
    df_scaled = dataScaler.minmax_scaler(df_mean_filled)
    CSV.save(df_scaled, "mean_minmax.csv")
    df_scaled = dataScaler.standardize_data(df_mean_filled)
    CSV.save(df_scaled, "mean_standardize.csv")

    df_median_filled = cleaner.fill_na_with_median()
    CSV.save(df_median_filled, "filled_with_median.csv")
    df_scaled = dataScaler.minmax_scaler(df_median_filled)
    CSV.save(df_median_filled, "median_minmax.csv")
    df_scaled = dataScaler.standardize_data(df_median_filled)
    CSV.save(df_scaled, "median_standardize.csv")


