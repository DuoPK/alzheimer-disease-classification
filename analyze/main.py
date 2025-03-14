from analyze.analyzer import AlzheimerDatasetAnalyzer

selected_features = [
    "Age", "Gender", "EducationLevel", "BMI", "Smoking", "AlcoholConsumption",
    "PhysicalActivity", "DietQuality", "SleepQuality", "FamilyHistoryAlzheimers",
    "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension",
    "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "MemoryComplaints",
    "BehavioralProblems", "ADL", "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness", "Diagnosis",  # Target variable
    "Ethnicity"  # One-hot encoded
]

file_path = "../data_generation/modified_extreme_dataset.csv"
analyzer = AlzheimerDatasetAnalyzer(file_path, selected_features)
analyzer.full_analysis(incorrect_data_to_nan=True, save_img=True, hist_bins=20)

file_path = "ready_dataset_with_nulls_as_incorrect_data.csv"
analyzer = AlzheimerDatasetAnalyzer(file_path, selected_features)
analyzer.full_analysis(incorrect_data_to_nan=False, save_img=False, hist_bins=20)
