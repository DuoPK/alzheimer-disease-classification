import math
from os import makedirs
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AlzheimerDatasetAnalyzer:
    def __init__(self, file_path, selected_features):
        """
        Initializes the dataset analyzer by loading data from a CSV file.

        Parameters:
        file_path (str): Path to the dataset CSV file.
        """
        self.selected_features = selected_features
        self.df = pd.read_csv(file_path)
        self.df = self.df[self.selected_features]

    def one_hot_encode_ethnicity(self):
        """
        Converts the 'Ethnicity' categorical feature into multiple binary columns (0/1).
        Creates 'Caucasian', 'African_American', 'Asian', 'Other' columns.
        Drops the original 'Ethnicity' column.
        """
        # Map Ethnicity to category names
        ethnicity_mapping = {0: "Caucasian", 1: "African_American", 2: "Asian", 3: "Other"}
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
        # Numerical features with known ranges
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

        # Check numeric ranges and replace invalid values with NaN
        for feature, (min_val, max_val) in valid_ranges.items():
            if feature in self.df.columns:
                mask = ~self.df[feature].between(min_val, max_val)
                if mask.any():
                    issues[
                        feature] = f"Values out of range (expected: {min_val}-{max_val}).\n{self.df.loc[mask, feature]}"
                    if change_to_nan:
                        self.df.loc[mask, feature] = np.nan

        # Check categorical/binary variables (expected values: 0 or 1)
        binary_features = [
            "Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
            "Diabetes", "Depression", "HeadInjury", "Hypertension", "MemoryComplaints",
            "BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges",
            "DifficultyCompletingTasks", "Forgetfulness", "Caucasian", "African_American",
            "Asian", "Other"
        ]

        for feature in binary_features:
            if feature in self.df.columns:
                mask = ~self.df[feature].isin([0, 1])
                if mask.any():
                    issues[feature] = f"Contains invalid values (expected: 0 or 1).\n{self.df.loc[mask, feature]}"
                    if change_to_nan:
                        self.df.loc[mask, feature] = np.nan

        # Check categorical features with fixed values
        categorical_features = {
            "EducationLevel": [0, 1, 2, 3],  # Valid levels: None, High School, Bachelor's, Higher
            "Ethnicity": [0, 1, 2, 3]  # {0: "Caucasian", 1: "African_American", 2: "Asian", 3: "Other"}
        }

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

    def full_analysis(self, incorrect_data_to_nan=False, save_img=False, hist_bins=20, show_count_on_hist=False,
                      path_imgs=""):
        print("\nRunning Full Analysis...\n")
        self.analyze_classes()
        self.check_missing_values()
        self.dataset_statistics()
        self.plot_histograms(bins=hist_bins, save_img=save_img, show_count=show_count_on_hist, path_imgs=path_imgs)
        self.plot_boxplots(save_img=save_img, path_imgs=path_imgs)
        self.validate_data_ranges(change_to_nan=incorrect_data_to_nan)
        if incorrect_data_to_nan:
            self.df.to_csv("ready_dataset_with_nulls_as_incorrect_data.csv", index=False)
        # if "Ethnicity" in self.df.columns:
        #     self.one_hot_encode_ethnicity()
        # self.df.to_csv("ready_dataset_with_nulls_as_incorrect_data_onehot.csv", index=False)
        print("\nFull analysis completed.")
