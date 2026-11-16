"""
Unit tests for core/report.py
"""
import json
import pytest
from core.report import generate_fhir_bundle, generate_pdf_report


@pytest.fixture
def mock_data():
    patient = {
        "male": 1,
        "age": 55.0,
        "currentSmoker": 1,
        "cigsPerDay": 20.0,
        "BPMeds": 1,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "totChol": 240.0,
        "sysBP": 150.0,
        "diaBP": 95.0,
        "BMI": 28.5,
        "heartRate": 80.0,
        "glucose": 115.0,
    }
    prediction = {
        "p_chd": 0.185,
        "p_chd_std": 0.021,
        "p_diabetes": 0.12,
        "p_diabetes_std": 0.015,
        "heart_age": 68.5,
        "heart_age_delta": 13.5,
        "life_expectancy": 71.2,
        "age": 55.0,
        "sex": 1,
    }
    recommendations = {
        "top_interventions": [{
            "id": "smoking_cessation",
            "title_key": "rec_smoking_title",
            "category": "lifestyle",
            "source": "WHO / ESC 2021",
            "rationale_key": "rec_smoking_rationale",
            "action_key": "rec_smoking_action",
            "delta_chd_pp": -4.2,
            "delta_heart_age": -5.1,
            "delta_life_exp": 1.5,
            "delta_diabetes_pp": 0.0,
        }]
    }
    return patient, prediction, recommendations


def test_fhir_bundle_has_hdl_only_when_entered(mock_data):
    patient, prediction, _ = mock_data

    def loinc(bundle):
        return {c["code"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"
                for c in e["resource"]["code"]["coding"]}

    assert "2085-9" not in loinc(generate_fhir_bundle(patient, prediction))
    assert "2085-9" in loinc(generate_fhir_bundle({**patient, "HDL": 52.0}, prediction))
    # resting heart rate is no longer collected, so it is not exported
    assert "8867-4" not in loinc(generate_fhir_bundle(patient, prediction))


def test_generate_fhir_bundle(mock_data):
    patient, prediction, _ = mock_data
    bundle = generate_fhir_bundle(patient, prediction, lang="ru")

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(bundle["entry"]) >= 7  # Patient + 5 observations + RiskAssessment

    # Validate json serializability
    json_str = json.dumps(bundle)
    assert len(json_str) > 0

    # Check LOINC codes
    loinc_codes = []
    for e in bundle["entry"]:
        res = e["resource"]
        if res["resourceType"] == "Observation":
            codings = res["code"]["coding"]
            for c in codings:
                if c["system"] == "http://loinc.org":
                    loinc_codes.append(c["code"])

    assert "8480-6" in loinc_codes  # SBP
    assert "8462-4" in loinc_codes  # DBP
    assert "2093-3" in loinc_codes  # Cholesterol
    assert "1558-6" in loinc_codes  # Fasting glucose
    assert "39156-5" in loinc_codes  # BMI


def test_generate_pdf_report(mock_data):
    patient, prediction, recs = mock_data
    pdf_bytes = generate_pdf_report(patient, prediction, recs, lang="ru", units="mmol")

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 5000
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_pdf_report_all_languages(mock_data):
    patient, prediction, recs = mock_data
    for lang in ["ru", "uz", "en"]:
        pdf_bytes = generate_pdf_report(patient, prediction, recs, lang=lang, units="mmol")
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(pdf_bytes) > 5000
