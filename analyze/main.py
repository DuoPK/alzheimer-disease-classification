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
analyzer.full_analysis()
