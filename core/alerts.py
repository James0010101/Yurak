"""
core/alerts.py — Findings a doctor would act on regardless of the 10-year risk.

A risk score can be "intermediate" while one value on its own already needs a
doctor soon. These checks run on the raw inputs (mg/dL, mmHg) and return
i18n keys, most urgent first.

Thresholds:
- BP >= 180 and/or >= 110 mmHg: grade 3 hypertension (ESC/ESH 2018).
- Fasting glucose >= 7.0 mmol/L (126 mg/dL) without a diagnosis: diabetic range,
  needs confirmation (WHO).
- Total cholesterol >= 7.5 mmol/L (290 mg/dL): very high; NCEP ATP III "high" starts
  at 6.2, this is well beyond it and worth a prompt lipid work-up.
- BMI >= 40: obesity class III (WHO).
"""
from typing import Any, Dict, List

URGENT, SOON = "urgent", "soon"


def find_alerts(patient: Dict[str, Any]) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []
    sbp = float(patient.get("sysBP", 120))
    dbp = float(patient.get("diaBP", 80))
    if sbp >= 180 or dbp >= 110:
        alerts.append({"level": URGENT, "key": "alert_bp_severe"})

    glucose = patient.get("glucose")
    if glucose is not None and float(glucose) >= 126 and not int(patient.get("diabetes", 0)):
        alerts.append({"level": SOON, "key": "alert_glucose_diabetic"})

    chol = patient.get("totChol")
    if chol is not None and float(chol) >= 290:
        alerts.append({"level": SOON, "key": "alert_chol_very_high"})

    if float(patient.get("BMI", 25)) >= 40:
        alerts.append({"level": SOON, "key": "alert_bmi_severe"})
    return alerts


def next_step_key(p_chd_percent: float, has_urgent: bool) -> str:
    """Plain-language 'what to do now', by risk band (NCEP ATP III 10/20%)."""
    if has_urgent:
        return "next_urgent"
    if p_chd_percent >= 20:
        return "next_high"
    if p_chd_percent >= 10:
        return "next_mid"
    return "next_low"
