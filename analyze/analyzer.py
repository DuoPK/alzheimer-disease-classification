import math
from os import makedirs
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from features import range_features, binary_features, categorical_features


class AlzheimerDatasetAnalyzer:
    def __init__(self, file_path, selected_features, dependent_feature):
        """
        Initializes the dataset analyzer by loading data from a CSV file.

        Parameters:
        file_path (str): Path to the dataset CSV file.
        """
        self.selected_features = selected_features
        self.dependent_feature = dependent_feature
        self.df = pd.read_csv(file_path)
        self.df = self.df[self.df.columns.intersection(self.selected_features)]

    def one_hot_encode_ethnicity(self):
        """
        Converts the 'Ethnicity' categorical feature into multiple binary columns (0/1).
        Creates 'Caucasian', 'African_American', 'Asian', 'Other' columns.
        Drops the original 'Ethnicity' column.
        """
        # Map Ethnicity to category names
        ethnicity_mapping = {0: "Caucasian", 1: "African_American", 2: "Asian", 3: "Other_ethnicity"}
        self.df["Ethnicity"] = self.df["Ethnicity"].map(ethnicity_mapping)

        # Apply one-hot encoding and convert to 0/1
        ethnicity_encoded = pd.get_dummies(self.df["Ethnicity"], prefix="", prefix_sep="").astype(int)

        # Merge with original DataFrame and drop 'Ethnicity'
        self.df = pd.concat([self.df.drop(columns=["Ethnicity"]), ethnicity_encoded], axis=1)

        print("\nOne-hot encoding for Ethnicity completed. New columns added:", list(ethnicity_encoded.columns))

    def analyze_classes(self):
        """
        Analyzes the distribution of classes (healthy vs. diagnosed with Alzheimer's).
        Displays the count of records per class.
        """
        class_counts = self.df["Diagnosis"].value_counts()
        print("\nClass Distribution:\n", class_counts)

        plt.figure(figsize=(6, 4))
        sns.barplot(x=class_counts.index, y=class_counts.values, palette="coolwarm", hue=class_counts.index)
        plt.xlabel("Diagnoza (0 = Zdrowy, 1 = Chory)")
        plt.ylabel("Liczba pacjentów")
        plt.title("Rozłożenie klas")
        plt.tight_layout()
        plt.savefig("class_distribution.jpg")
        plt.show()

    def check_missing_values(self):
        """
        Checks for missing values in the dataset.
        Displays the number of missing values per feature.
        """
        missing_values = self.df.isnull().sum()
        print("\nMissing values in the dataset:")
        print(missing_values[missing_values > 0])

    def dataset_statistics(self):
        """
        Prints basic descriptive statistics for the selected numerical features.
        """
        print("\nDataset statistics:")
        print(self.df.describe())

    def validate_data_ranges(self, change_to_nan=False):
        """
        Validates data ranges for selected features and reports anomalies.
        """
        issues = {}

        # Check numeric ranges
        for feature, (min_val, max_val) in range_features.items():
            if feature in self.df.columns:
                mask = ~self.df[feature].between(min_val, max_val)
                if mask.any():
                    issues[
                        feature] = f"Values out of range (expected: {min_val}-{max_val}).\n{self.df.loc[mask, feature]}"
                    if change_to_nan:
                        self.df.loc[mask, feature] = np.nan

        # Check categorical/binary variables (expected values: 0 or 1)
        for feature in binary_features:
            if feature in self.df.columns:
                mask = ~self.df[feature].isin([0, 1])
                if mask.any():
                    issues[feature] = f"Contains invalid values (expected: 0 or 1).\n{self.df.loc[mask, feature]}"
                    if change_to_nan:
                        self.df.loc[mask, feature] = np.nan

        # Check categorical features with fixed values
        for feature, valid_values in categorical_features.items():
            if feature in self.df.columns:
                mask = ~self.df[feature].isin(valid_values)
                if mask.any():
                    issues[
                        feature] = f"Contains invalid values (expected: {valid_values}).\n{self.df.loc[mask, feature]}"
                    if change_to_nan:
                        self.df.loc[mask, feature] = np.nan

        # Display detected issues
        if issues:
            print("\n Data Range Issues Detected:")
            for key, value in issues.items():
                print(f"- {key}: {value}")
        else:
            print("\n All values are within expected ranges.")

    def plot_histograms(self, bins=20, save_img=False, show_count=False, path_imgs=""):
        """
        Plots histograms for all selected features.
        The histograms are saved into JPEG files with a maximum layout of 1 row x 3 columns per file.

        :param bins: Number of bins in histograms.
        :param save_img: If True, saves the images as JPEG files.
        :param show_count: If True, displays sample count above each bin.
        """
        features = self.df.columns
        num_features = len(features)
        plots_per_fig = 3
        num_figs = math.ceil(num_features / plots_per_fig)

        for fig_idx in range(num_figs):
            start = fig_idx * plots_per_fig
            end = start + plots_per_fig
            batch_features = features[start:end]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.flatten()

            for i, feature in enumerate(batch_features):
                ax = axes[i]
                counts, bin_edges, patches = ax.hist(self.df[feature], bins=bins, color="skyblue", edgecolor="black")

                ax.set_title(feature)

                if show_count:
                    for count, bin_patch in zip(counts, patches):
                        if count > 0:
                            bin_x = bin_patch.get_x() + bin_patch.get_width() / 2
                            ax.text(bin_x, count, str(int(count)), ha='center', va='bottom', fontsize=10, color='black')

            # Remove any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            filename = Path(f"histograms_{fig_idx + 1}.jpg")
            if save_img:
                if path_imgs:
                    makedirs(path_imgs, exist_ok=True)
                plt.savefig(path_imgs / filename)
            plt.show()
            print(f"Histogram batch {fig_idx + 1} saved as {filename}")
            plt.close(fig)

    def plot_boxplots(self, save_img=False, path_imgs=""):
        """
        Generates box plots for continuous features only (features with >2 unique values).
        The box plots are saved into JPEG files with a maximum layout of 2 rows x 3 columns per file.
        """
        # Select continuous features: only those with more than 2 unique values.
        continuous_features = [feature for feature in self.df.columns if self.df[feature].nunique() > 2]
        num_features = len(continuous_features)
        plots_per_fig = 6
        num_figs = math.ceil(num_features / plots_per_fig)

        for fig_idx in range(num_figs):
            start = fig_idx * plots_per_fig
            end = start + plots_per_fig
            batch_features = continuous_features[start:end]

            fig, axes = plt.subplots(1, 6, figsize=(15, 5))
            axes = axes.flatten()

            for i, feature in enumerate(batch_features):
                sns.boxplot(y=self.df[feature], ax=axes[i])
                axes[i].set_title(feature)
                axes[i].set_xlabel("")

            # Remove unused axes
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # fig.suptitle("Box Plots", fontsize=16)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            filename = Path(f"boxplots_{fig_idx + 1}.jpg")
            if save_img:
                if path_imgs:
                    makedirs(path_imgs, exist_ok=True)
                plt.savefig(path_imgs / filename)
            plt.show()
            print(f"Box plot batch {fig_idx + 1} saved as {filename}")
            plt.close(fig)

    def plot_pairplots(self, selected_cols, save_img=False, path_imgs=""):
        """
        Generates a pairplot for selected features (default: cholesterol-related).

        Parameters:
        selected_cols (list): List of column names to include in the pairplot.
        save_img (bool): If True, saves the plot as a PNG file.
        path_imgs (str): Directory path where the image will be saved (if save_img=True).
        """

        available_cols = [col for col in selected_cols if col in self.df.columns]

        if len(available_cols) < 2:
            print("Not enough selected features available in the dataset for a pairplot.")
            return

        sns.pairplot(self.df[available_cols])
        plt.tight_layout()

        if save_img:
            filename = Path("pairplot_selected_variables.png")
            if path_imgs:
                makedirs(path_imgs, exist_ok=True)
                filename = Path(path_imgs) / filename
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Pairplot saved as {filename}")

        plt.show()
        plt.close()

    def plot_correlation_heatmap(self, method="spearman", save_img=False, path_imgs=""):
        """
        Plots a heatmap of correlation between numeric features using the specified method.

        Parameters:
        method (str): Correlation method ('pearson', 'spearman').
        save_img (bool): Whether to save the heatmap image.
        path_imgs (str): Directory to save the image if save_img is True.
        """
        numeric_cols = self.df.select_dtypes(include='number').columns
        correlation_matrix = self.df[numeric_cols].corr(method=method)

        plt.figure(figsize=(18, 14))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            cbar=True,
            square=True,
            linewidths=0.3,
            annot_kws={"size": 10},
            cbar_kws={"shrink": 0.8}
        )

        plt.title(f"{method.capitalize()} Correlation Heatmap", fontsize=16)
        plt.tight_layout()

        if save_img:
            filename = Path(f"correlation_heatmap_{method}.png")
            if path_imgs:
                makedirs(path_imgs, exist_ok=True)
                filename = Path(path_imgs) / filename
            plt.savefig(filename, dpi=1000, bbox_inches='tight')
            print(f"Correlation heatmap saved as {filename}")

        plt.show()
        plt.close()

    def remove_rows_with_missing_dependent_value(self):
        """
        Removes rows with missing values in the dependent feature.
        """
        print(f"Old dataset shape: {self.df.shape}")
        self.df = self.df.dropna(subset=[self.dependent_feature])
        print(f"\nRows with missing values in '{self.dependent_feature}' removed. New dataset shape: {self.df.shape}")

    def full_analysis(self, incorrect_data_to_nan=False, save_img=False, hist_bins=20, show_count_on_hist=False,
                      path_imgs="", remove_missing_dependent=False):
        print("\nRunning Full Analysis...\n")
        self.analyze_classes()
        self.check_missing_values()
        self.dataset_statistics()
        self.plot_histograms(bins=hist_bins, save_img=save_img, show_count=show_count_on_hist, path_imgs=path_imgs)
        self.plot_boxplots(save_img=save_img, path_imgs=path_imgs)
        self.validate_data_ranges(change_to_nan=incorrect_data_to_nan)
        self.plot_pairplots(selected_cols=['CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL'])
        self.plot_correlation_heatmap(method="spearman")
        if remove_missing_dependent:
            self.remove_rows_with_missing_dependent_value()
        if incorrect_data_to_nan or remove_missing_dependent:
            self.df.to_csv("ready_dataset_to_preprocessing.csv", index=False)
        print("\nFull analysis completed.")

    def check_and_save_dataset(self, incorrect_data_to_nan=True, remove_missing_dependent=True):
        self.validate_data_ranges(change_to_nan=incorrect_data_to_nan)
        if remove_missing_dependent:
            self.remove_rows_with_missing_dependent_value()
        if incorrect_data_to_nan or remove_missing_dependent:
            self.df.to_csv("../ready_dataset_to_preprocessing.csv", index=False)
