from features import range_features


class DataScaler:

    def minmax_scaler(self, df, new_max=1, new_min=0):
        numeric_cols = list(range_features.keys()) + ["EducationLevel"]
        df_scaled = df.copy()
        for col in numeric_cols:
            min_value = df_scaled[col].min()
            max_value = df_scaled[col].max()
            df_scaled[col] = ((df_scaled[col] - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min

        return df_scaled

    def standardize_data(self, df):
        numeric_cols = list(range_features.keys()) + ["EducationLevel"]
        df_standardized = df.copy()

        for col in numeric_cols:
            mean_value = df_standardized[col].mean()
            std_value = df_standardized[col].std()
            df_standardized[col] = (df_standardized[col] - mean_value) / std_value

        return df_standardized