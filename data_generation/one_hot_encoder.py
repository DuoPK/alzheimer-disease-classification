import pandas as pd


class OneHotEncoder:
    def one_hot_encode(self, df, feature, category_names, drop_original=True):
        """
        Converts the specified categorical feature into multiple binary columns (0/1).
        Creates new columns based on the unique categories in the feature.
        Drops the original feature column.
        """
        # Map Ethnicity to category names
        mapping = {i: name for i, name in enumerate(category_names)}
        df[feature] = df[feature].map(mapping)

        # Apply one-hot encoding and convert to 0/1
        encoded = pd.get_dummies(df[feature], prefix="", prefix_sep="").astype(int)

        # Merge with original DataFrame and drop the original feature
        if drop_original:
            df = pd.concat([df.drop(columns=[feature]), encoded], axis=1)
        else:
            df = pd.concat([df, encoded], axis=1)

        return df
