range_features = {
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

int_features = [
    "Age",
    # "AlcoholConsumption",
    # "PhysicalActivity",
    # "DietQuality",
    # "SleepQuality",
    # "MMSE",
    # "FunctionalAssessment",
    # "ADL"
]

binary_features = [
    "Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
    "Diabetes", "Depression", "HeadInjury", "Hypertension", "MemoryComplaints",
    "BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness", "Diagnosis",
    "Caucasian", "African_American", "Asian", "Other_ethnicity"
]

categorical_features = {
    "EducationLevel": [0, 1, 2, 3],  # Valid levels: None, High School, Bachelor's, Higher
    "Ethnicity": [0, 1, 2, 3]  # {0: "Caucasian", 1: "African_American", 2: "Asian", 3: "Other"}
}
