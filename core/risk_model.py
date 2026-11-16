"""
core/risk_model.py — Risk calculation, uncertainty, Integrated Gradients XAI,
and intervention simulation engine.
"""
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.clinical import (
    calculate_dagostino_risk,
    calculate_dagostino_risk_bmi,
    calculate_framingham_heart_age,
    calculate_heart_age_bmi,
    calculate_uzbekistan_life_expectancy,
)
from core.config import (
    DEFAULT_HDL_MGDL,
    DIABETES_GLUCOSE_MGDL,
    FEATURE_NAMES,
    FIXED_HEART_RATE,
    MODELS_DIR,
)
from neural_network import MultiTaskNeuralNetwork

logger = logging.getLogger(__name__)

# Default weights path
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "model_weights.npz"

# Cache resource wrapper for Streamlit if available
try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    def cache_decorator(func):
        return func


@cache_decorator
def load_risk_model(weights_path: Optional[Union[str, Path]] = None) -> MultiTaskNeuralNetwork:
    """
    Loads and caches the trained MultiTaskNeuralNetwork.
    """
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model weights file not found: {path}")
    logger.info(f"Loading risk model from {path}...")
    model = MultiTaskNeuralNetwork.load(path)
    return model


def patient_dict_to_raw_vector(patient: Dict[str, Any]) -> np.ndarray:
    """
    Converts a patient dictionary into a 1D NumPy array strictly matching FEATURE_NAMES.
    """
    raw_values = [
        float(patient.get("male", 1)),
        float(patient.get("age", 50.0)),
        float(patient.get("currentSmoker", 0)),
        float(patient.get("cigsPerDay", 0.0)),
        float(patient.get("BPMeds", 0)),
        float(patient.get("prevalentStroke", 0)),
        float(patient.get("prevalentHyp", 0)),
        float(patient.get("totChol", 200.0)),
        float(patient.get("sysBP", 120.0)),
        float(patient.get("diaBP", 80.0)),
        float(patient.get("BMI", 25.0)),
        float(patient.get("heartRate", FIXED_HEART_RATE)),
        float(patient.get("glucose", 90.0)),
    ]
    return np.array(raw_values, dtype=np.float64)


def get_optimal_healthy_profile(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs an age- and sex-matched reference profile with optimal clinical markers.
    Used as the baseline for Integrated Gradients and risk gap comparisons.
    """
    return {
        "male": int(patient.get("male", 1)),
        "age": float(patient.get("age", 50.0)),
        "currentSmoker": 0,
        "cigsPerDay": 0.0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 0,
        "totChol": 180.0,
        "sysBP": 115.0,
        "diaBP": 75.0,
        "BMI": 22.0,
        "heartRate": FIXED_HEART_RATE,
        "glucose": 85.0,
    }


def has_diabetes(patient: Dict[str, Any]) -> bool:
    """
    Diabetes for the heart-age formula: a known diagnosis OR fasting glucose
    >= 7.0 mmol/L (126 mg/dL). Previously the formula ignored glucose entirely,
    so an undiagnosed person with glucose 9 mmol/L got the heart age of a non-diabetic.
    """
    known = int(patient.get("diabetes", 0)) == 1
    glucose = patient.get("glucose", 90.0)
    return known or (glucose is not None and float(glucose) >= DIABETES_GLUCOSE_MGDL)


# Lab values a person may not know. When missing, the network gets the training
# median (exactly how missing values were imputed during training) and heart age
# switches to the D'Agostino BMI model, which needs no blood test.
OPTIONAL_LABS = ("totChol", "glucose")


def fill_missing_labs(patient: Dict[str, Any], model: MultiTaskNeuralNetwork) -> Dict[str, Any]:
    meta = model.metadata or {}
    medians = dict(zip(meta.get("feature_names", FEATURE_NAMES), meta.get("medians", [])))
    filled = dict(patient)
    for k in OPTIONAL_LABS:
        if filled.get(k) is None and k in medians:
            filled[k] = float(medians[k])
    return filled


def normalize_features(
    raw_vector: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    Applies z-score standardization using training set statistics.
    """
    stds_safe = np.where(stds == 0, 1.0, stds)
    return (raw_vector - means) / stds_safe


def predict_patient(
    patient: Dict[str, Any],
    model: Optional[MultiTaskNeuralNetwork] = None,
    n_mc: int = 0
) -> Dict[str, Any]:
    """
    Full clinical evaluation:
    - 10-year CHD risk (neural network)
    - Diabetes screening probability (neural network)
    - Heart age (D'Agostino 2008: lipid model, or BMI model when cholesterol is unknown)
    - Life expectancy (WHO Uzbekistan life table read at heart age)
    - General CVD risk (D'Agostino 2008 benchmark)

    MC Dropout uncertainty is only computed when n_mc > 0: the UI no longer shows
    it, and 30 extra forward passes per simulation made every rerun slow.
    """
    if model is None:
        model = load_risk_model()

    chol_known = patient.get("totChol") is not None
    glucose_known = patient.get("glucose") is not None
    filled = fill_missing_labs(patient, model)

    meta = model.metadata or {}
    means = np.array(meta.get("means", [0.0] * len(FEATURE_NAMES)), dtype=np.float64)
    stds = np.array(meta.get("stds", [1.0] * len(FEATURE_NAMES)), dtype=np.float64)

    raw_vec = patient_dict_to_raw_vector(filled)
    norm_vec = normalize_features(raw_vec, means, stds).reshape(1, -1)

    probs = model.predict(norm_vec)
    p_chd = float(probs["chd"][0, 0])
    p_dm = float(probs["diabetes"][0, 0])

    chd_std = dm_std = 0.0
    if n_mc > 0:
        unc = model.predict_with_uncertainty(norm_vec, n_mc=n_mc, seed=42)
        chd_std = float(unc["chd"]["std"][0, 0])
        dm_std = float(unc["diabetes"]["std"][0, 0])

    age = float(patient.get("age", 50.0))
    sex = int(patient.get("male", 1))
    sbp = float(patient.get("sysBP", 120.0))
    bp_meds = int(patient.get("BPMeds", 0))
    smoker = int(patient.get("currentSmoker", 0))
    hdl = patient.get("HDL")
    hdl = float(hdl) if hdl else DEFAULT_HDL_MGDL
    diabetes_flag = int(has_diabetes(patient))

    if chol_known:
        tot_chol = float(patient["totChol"])
        kw = dict(age=age, sex=sex, sbp=sbp, bp_meds=bp_meds, tot_chol=tot_chol, hdl=hdl,
                  smoker=smoker, diabetes=diabetes_flag)
        heart_age = calculate_framingham_heart_age(**kw)
        dagostino_cvd = calculate_dagostino_risk(**kw)
        method = "lab"
    else:
        bmi = float(patient.get("BMI", 25.0))
        kw = dict(age=age, sex=sex, sbp=sbp, bp_meds=bp_meds, bmi=bmi, smoker=smoker, diabetes=diabetes_flag)
        heart_age = calculate_heart_age_bmi(**kw)
        dagostino_cvd = calculate_dagostino_risk_bmi(**kw)
        method = "bmi"

    life_exp = calculate_uzbekistan_life_expectancy(age=age, sex=sex, heart_age=heart_age)

    return {
        "heart_age_method": method,
        "glucose_known": glucose_known,
        "diabetes_flag": diabetes_flag,
        "p_chd": round(p_chd, 4),
        "p_chd_std": round(chd_std, 4),
        "p_diabetes": round(p_dm, 4),
        "p_diabetes_std": round(dm_std, 4),
        "heart_age": round(heart_age, 1),
        "heart_age_delta": round(heart_age - age, 1),
        "life_expectancy": round(life_exp, 1),
        "dagostino_cvd_risk": round(dagostino_cvd, 4),
        "age": age,
        "sex": sex,
    }


def explain_prediction(
    patient: Dict[str, Any],
    head: str = "chd",
    m_steps: int = 50,
    model: Optional[MultiTaskNeuralNetwork] = None
) -> Dict[str, Any]:
    """
    Computes feature attributions via Integrated Gradients starting from an
    age- and sex-matched healthy baseline profile.
    Attributions are scaled to percentage points (pp).
    """
    if model is None:
        model = load_risk_model()
    missing = [k for k in OPTIONAL_LABS if patient.get(k) is None]
    patient = fill_missing_labs(patient, model)

    meta = model.metadata or {}
    means = np.array(meta.get("means", [0.0] * len(FEATURE_NAMES)), dtype=np.float64)
    stds = np.array(meta.get("stds", [1.0] * len(FEATURE_NAMES)), dtype=np.float64)

    raw_patient = patient_dict_to_raw_vector(patient)
    norm_patient = normalize_features(raw_patient, means, stds)

    healthy_ref = get_optimal_healthy_profile(patient)
    # An unknown lab value is not a risk factor of this person: give the reference
    # the same imputed value so its attribution is exactly zero.
    for k in missing:
        healthy_ref[k] = patient[k]
    raw_base = patient_dict_to_raw_vector(healthy_ref)
    norm_base = normalize_features(raw_base, means, stds)

    # Integrated Gradients
    raw_attributions = model.integrated_gradients(
        norm_patient,
        norm_base,
        head=head,
        m_steps=m_steps,
    )

    p_patient = float(model.predict(norm_patient.reshape(1, -1))[head][0, 0])
    p_base = float(model.predict(norm_base.reshape(1, -1))[head][0, 0])

    attributions_pp = {}
    for i, name in enumerate(FEATURE_NAMES):
        attributions_pp[name] = round(float(raw_attributions[i] * 100.0), 2)

    sum_pp = round(float(np.sum(raw_attributions) * 100.0), 2)
    diff_pp = round(float((p_patient - p_base) * 100.0), 2)

    return {
        "head": head,
        "attributions_pp": attributions_pp,
        "patient_prob": round(p_patient, 4),
        "baseline_prob": round(p_base, 4),
        "sum_attributions_pp": sum_pp,
        "target_diff_pp": diff_pp,
    }


# Correlated inputs describing one clinical factor are shown as a single bar.
# Integrated Gradients is additive, so summing their attributions is exact.
ATTRIBUTION_GROUPS: Dict[str, tuple] = {"smoking": ("currentSmoker", "cigsPerDay")}


def group_attributions(attributions_pp: Dict[str, float]) -> Dict[str, float]:
    """Merge correlated features (smoker flag + cigarettes/day) into one factor."""
    grouped = dict(attributions_pp)
    for group, members in ATTRIBUTION_GROUPS.items():
        present = [m for m in members if m in grouped]
        if present:
            grouped[group] = round(sum(grouped.pop(m) for m in present), 2)
    return grouped


def simulate_intervention(
    patient: Dict[str, Any],
    changes: Dict[str, Any],
    model: Optional[MultiTaskNeuralNetwork] = None
) -> Dict[str, Any]:
    """
    Simulates the quantitative impact of clinical interventions on CHD risk,
    heart age, life expectancy, and diabetes screening probability.
    """
    if model is None:
        model = load_risk_model()

    base_eval = predict_patient(patient, model=model)

    sim_patient = deepcopy(patient)
    sim_patient.update(changes)

    # If currentSmoker set to 0, automatically zero out cigsPerDay unless specified
    if changes.get("currentSmoker") == 0 and "cigsPerDay" not in changes:
        sim_patient["cigsPerDay"] = 0.0

    sim_eval = predict_patient(sim_patient, model=model)

    delta_chd_pp = round((sim_eval["p_chd"] - base_eval["p_chd"]) * 100.0, 2)
    delta_heart_age = round(sim_eval["heart_age"] - base_eval["heart_age"], 1)
    delta_life_exp = round(sim_eval["life_expectancy"] - base_eval["life_expectancy"], 1)
    delta_dm_pp = round((sim_eval["p_diabetes"] - base_eval["p_diabetes"]) * 100.0, 2)

    return {
        "base": base_eval,
        "simulated": sim_eval,
        "deltas": {
            "delta_chd_pp": delta_chd_pp,
            "delta_heart_age": delta_heart_age,
            "delta_life_exp": delta_life_exp,
            "delta_diabetes_pp": delta_dm_pp,
        },
    }


def compute_risk_trajectory(
    patient: Dict[str, Any],
    years_ahead: Tuple[int, ...] = (0, 5, 10, 15, 20),
    improved_changes: Optional[Dict[str, Any]] = None,
    model: Optional[MultiTaskNeuralNetwork] = None
) -> List[Dict[str, Any]]:
    """
    Computes risk trajectories across future years under:
    1) Current trajectory (only age changes; all other markers stay as entered)
    2) Optimized trajectory (with beneficial interventions applied)

    No physiological drift is assumed: an earlier version added +0.4 mmHg SBP per
    year, a number without a verified source. Ages above 70 are outside the
    training cohort and are flagged with `extrapolated`.
    """
    if model is None:
        model = load_risk_model()

    base_age = float(patient.get("age", 50.0))
    trajectory = []

    for y in years_ahead:
        sim_age = min(85.0, base_age + y)

        # 1. Current scenario
        curr_p = deepcopy(patient)
        curr_p["age"] = sim_age

        res_curr = predict_patient(curr_p, model=model)
        trajectory.append({
            "years_ahead": y,
            "age": sim_age,
            "extrapolated": sim_age > 70.0,
            "scenario": "current",
            "p_chd_percent": round(res_curr["p_chd"] * 100.0, 1),
            "heart_age": res_curr["heart_age"],
            "life_expectancy": res_curr["life_expectancy"],
        })

        # 2. Optimized scenario (if changes provided)
        if improved_changes:
            opt_p = deepcopy(patient)
            opt_p.update(improved_changes)
            opt_p["age"] = sim_age
            if opt_p.get("currentSmoker") == 0:
                opt_p["cigsPerDay"] = 0.0

            res_opt = predict_patient(opt_p, model=model)
            trajectory.append({
                "years_ahead": y,
                "age": sim_age,
                "extrapolated": sim_age > 70.0,
                "scenario": "improved",
                "p_chd_percent": round(res_opt["p_chd"] * 100.0, 1),
                "heart_age": res_opt["heart_age"],
                "life_expectancy": res_opt["life_expectancy"],
            })

    return trajectory
