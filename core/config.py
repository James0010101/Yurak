"""
Configuration constants and paths for Yurak.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
I18N_DIR = BASE_DIR / "i18n"
DOCS_DIR = BASE_DIR / "docs"

# 13 features used by the neural network (strict ordering)
FEATURE_NAMES = [
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

# Classification heads
HEAD_NAMES = ["chd", "diabetes"]

# Data training boundaries (Framingham cohort range)
MIN_TRAIN_AGE = 32
MAX_TRAIN_AGE = 70

# Typical 1st to 99th percentiles from Framingham dataset for UI validation warnings
FEATURE_PERCENTILES_1_99 = {
    "sysBP": (97.0, 200.0),
    "diaBP": (60.0, 118.0),
    "totChol": (153.0, 354.0),
    "BMI": (18.2, 39.0),
    "glucose": (55.0, 175.0),
    "heartRate": (52.0, 110.0),
    "cigsPerDay": (0.0, 43.0),
}

# Clinical reference defaults (in mg/dL for lab values)
DEFAULT_HDL_MGDL = 45.0  # ~1.16 mmol/L, used when the person does not know their HDL

# Resting heart rate is a model input, but its effect on the trained network is
# negligible (≈1.5 pp across the 60–95 bpm range, with the wrong sign), so the
# app no longer asks for it. It is fixed at the healthy-reference value, which
# also makes its Integrated Gradients attribution exactly zero.
FIXED_HEART_RATE = 70.0

# Fasting glucose at or above this value (mg/dL, = 7.0 mmol/L) counts as diabetes
# for the D'Agostino heart-age formula even without a prior diagnosis (WHO criterion).
DIABETES_GLUCOSE_MGDL = 126.0
REFERENCE_OPTIMAL_PROFILE = {
    "sysBP": 120.0,
    "diaBP": 80.0,
    "totChol": 180.0,
    "HDL": 45.0,
    "BMI": 22.0,
    "glucose": 90.0,
    "heartRate": 70.0,
    "currentSmoker": 0,
    "cigsPerDay": 0,
    "BPMeds": 0,
    "prevalentStroke": 0,
    "prevalentHyp": 0,
}
