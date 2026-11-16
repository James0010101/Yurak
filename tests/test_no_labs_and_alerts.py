"""
Behaviour added for people without blood tests, red-flag alerts and the
corrected D'Agostino constant.
"""
import pytest

from core.alerts import URGENT, find_alerts, next_step_key
from core.clinical import (
    DAGOSTINO_COEFFS,
    calculate_dagostino_risk_bmi,
    calculate_heart_age_bmi,
)
from core.recommendations import generate_recommendations
from core.report import generate_fhir_bundle, generate_pdf_report
from core.risk_model import explain_prediction, predict_patient

BASE = {"male": 1, "age": 55.0, "currentSmoker": 1, "cigsPerDay": 20.0, "BPMeds": 0, "prevalentStroke": 0,
        "prevalentHyp": 1, "totChol": 240.0, "sysBP": 150.0, "diaBP": 95.0, "BMI": 28.5, "heartRate": 70.0,
        "glucose": 115.0, "diabetes": 0}


def test_lab_model_male_constant_matches_framingham_page():
    assert DAGOSTINO_COEFFS["male"]["l_mean"] == pytest.approx(23.9802)


@pytest.mark.parametrize("sex", [0, 1])
def test_bmi_heart_age_equals_age_for_reference_person(sex):
    # Reference of the Framingham BMI calculator: SBP 125 untreated, BMI 22.5, no smoking, no diabetes
    assert calculate_heart_age_bmi(50, sex, 125, 0, 22.5, 0, 0) == pytest.approx(50.0, abs=0.01)


def test_bmi_model_monotonic_and_bounded():
    lo = calculate_heart_age_bmi(50, 1, 120, 0, 22, 0, 0)
    hi = calculate_heart_age_bmi(50, 1, 160, 0, 32, 1, 1)
    assert hi > lo
    assert 0.0 < calculate_dagostino_risk_bmi(50, 1, 160, 0, 32, 1, 1) < 1.0


def test_prediction_without_any_blood_test():
    p = {**BASE, "totChol": None, "glucose": None, "HDL": None}
    res = predict_patient(p)
    assert res["heart_age_method"] == "bmi"
    assert res["glucose_known"] is False
    assert 0.0 <= res["p_chd"] <= 1.0
    # Unknown labs are not this person's risk factors: zero attribution
    attrs = explain_prediction(p)["attributions_pp"]
    assert attrs["totChol"] == 0.0 and attrs["glucose"] == 0.0
    # No cholesterol/glucose step can be suggested from values nobody entered
    ids = {r["id"] for r in generate_recommendations(p)["all_interventions"]}
    assert "lipid_management" not in ids and "glycemic_control" not in ids


def test_report_and_fhir_without_labs():
    p = {**BASE, "totChol": None, "glucose": None}
    pred = predict_patient(p)
    recs = generate_recommendations(p)
    assert generate_pdf_report(p, pred, recs, lang="uz").startswith(b"%PDF-")
    codes = {c["code"] for e in generate_fhir_bundle(p, pred)["entry"]
             if e["resource"]["resourceType"] == "Observation" for c in e["resource"]["code"]["coding"]}
    assert "2093-3" not in codes and "1558-6" not in codes  # imputed values are never exported


def test_weight_goal_is_realistic():
    recs = generate_recommendations({**BASE, "BMI": 34.0})
    weight = next(r for r in recs["all_interventions"] if r["id"] == "weight_reduction")
    assert weight["changes"]["BMI"] == pytest.approx(30.6)  # -10%, not straight to 24.5


def test_alerts():
    assert find_alerts({**BASE, "sysBP": 185})[0]["level"] == URGENT
    keys = {a["key"] for a in find_alerts({**BASE, "glucose": 130, "totChol": 300, "BMI": 41})}
    assert keys == {"alert_glucose_diabetic", "alert_chol_very_high", "alert_bmi_severe"}
    assert find_alerts({**BASE, "glucose": 130, "diabetes": 1}) == []  # already diagnosed
    assert find_alerts({**BASE, "glucose": None, "totChol": None}) == []
    assert next_step_key(5, False) == "next_low"
    assert next_step_key(15, False) == "next_mid"
    assert next_step_key(25, False) == "next_high"
    assert next_step_key(5, True) == "next_urgent"
