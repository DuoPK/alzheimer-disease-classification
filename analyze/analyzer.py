import math

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

    def preprocess_data(self):
        """
        Runs all preprocessing steps
        """
        if "Ethnicity" in self.df.columns:
            self.one_hot_encode_ethnicity()
        self.validate_data_ranges()

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

        # Update selected features after one-hot encoding
        # self.selected_features = [feature for feature in self.selected_features if feature != "Ethnicity"] + list(ethnicity_encoded.columns)

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
        plt.xlabel("Diagnosis (0 = No Alzheimer’s, 1 = Alzheimer’s)")
        plt.ylabel("Number of Patients")
        plt.title("Class Distribution in Dataset")
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

    def validate_data_ranges(self):
        """
        Validates data ranges for selected features and reports anomalies.
        """
        issues = {}
        # Numerical features with known ranges
        valid_ranges = {
            "Age": (60, 90),
            "BMI": (15, 40),
            "AlcoholConsumption": (0, 20),
            "PhysicalActivity": (0, 10),
            "DietQuality": (0, 10),
            "SleepQuality": (4, 10),
            "SystolicBP": (90, 180),
            "DiastolicBP": (60, 120),
            "CholesterolTotal": (150, 300),
            "CholesterolLDL": (50, 200),
            "CholesterolHDL": (20, 100),
            "CholesterolTriglycerides": (50, 400),
            "MMSE": (0, 30),
            "FunctionalAssessment": (0, 10),
            "ADL": (0, 10)
        }

        # Check numeric ranges and replace invalid values with NaN
        for feature, (min_val, max_val) in valid_ranges.items():
            if feature in self.df.columns:
                mask = ~self.df[feature].between(min_val, max_val)
                if mask.any():
                    issues[feature] = f"Values out of range (expected: {min_val}-{max_val})."
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
                    issues[feature] = f"Contains invalid values (expected: 0 or 1)."
                    self.df.loc[mask, feature] = np.nan

        # Check categorical features with fixed values
        categorical_features = {
            "EducationLevel": [0, 1, 2, 3]  # Valid levels: None, High School, Bachelor's, Higher
        }

        for feature, valid_values in categorical_features.items():
            if feature in self.df.columns:
                mask = ~self.df[feature].isin(valid_values)
                if mask.any():
                    issues[feature] = f"Contains invalid values (expected: {valid_values})."
                    self.df.loc[mask, feature] = np.nan

        # Display detected issues
        if issues:
            print("\n Data Range Issues Detected:")
            for key, value in issues.items():
                print(f"- {key}: {value}")
        else:
            print("\n All values are within expected ranges.")

    def plot_histograms(self, bins=20):
        """
        Plots histograms for all selected features.
        The histograms are saved into JPEG files with a maximum layout of 2 rows x 3 columns per file.
        :param bins: Number of bins in histograms.
        """
        features = self.df.columns
        num_features = len(features)
        plots_per_fig = 6  # 2 rows x 3 columns
        num_figs = math.ceil(num_features / plots_per_fig)

        for fig_idx in range(num_figs):
            start = fig_idx * plots_per_fig
            end = start + plots_per_fig
            batch_features = features[start:end]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, feature in enumerate(batch_features):
                ax = axes[i]
                self.df[feature].hist(bins=bins, ax=ax, color="skyblue", edgecolor="black")
                ax.set_title(feature)

            # Remove any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle("Histograms of Selected Features", fontsize=16)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            filename = f"histograms_{fig_idx + 1}.jpg"
            plt.savefig(filename)
            plt.show()
            print(f"Histogram batch {fig_idx + 1} saved as {filename}")
            plt.close(fig)

    def plot_boxplots(self):
        """
        Generates box plots for continuous features only (features with >2 unique values).
        The box plots are saved into JPEG files with a maximum layout of 2 rows x 3 columns per file.
        """
        # Select continuous features: only those with more than 2 unique values.
        continuous_features = [feature for feature in self.df.columns if self.df[feature].nunique() > 2]
        num_features = len(continuous_features)
        plots_per_fig = 6  # 2 rows x 3 columns
        num_figs = math.ceil(num_features / plots_per_fig)

        for fig_idx in range(num_figs):
            start = fig_idx * plots_per_fig
            end = start + plots_per_fig
            batch_features = continuous_features[start:end]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, feature in enumerate(batch_features):
                sns.boxplot(y=self.df[feature], ax=axes[i])
                axes[i].set_title(feature)
                axes[i].set_xlabel("")

            # Remove unused axes
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle("Box Plots of Continuous Features", fontsize=16)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            filename = f"boxplots_{fig_idx + 1}.jpg"
            plt.savefig(filename)
            plt.show()
            print(f"Box plot batch {fig_idx + 1} saved as {filename}")
            plt.close(fig)

    def full_analysis(self):
        print("\nRunning Full Analysis...\n")
        self.preprocess_data()
        self.analyze_classes()
        self.check_missing_values()
        self.dataset_statistics()
        self.plot_histograms()
        self.plot_boxplots()
        print("\nFull analysis completed.")
