"""
app_pages/screening.py — Yurak: the screening form and its results.

Left column: a form with only the inputs that change a result.
Right column: heart-age headline, three result cards, ranked steps you can
switch on and off, and tabs with the details (factors, markers, 20-year view, report).
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

from core.branding import BRAND_ANOR, BRAND_TEAL, logo_header_html
from core.clinical import calculate_life_expectancy_estimate
from core.config import FIXED_HEART_RATE, MAX_TRAIN_AGE, MIN_TRAIN_AGE
from core.i18n import get_current_language, t
from core.recommendations import generate_recommendations
from core.report import generate_fhir_bundle, generate_pdf_report
from core.risk_model import (
    compute_risk_trajectory,
    explain_prediction,
    group_attributions,
    predict_patient,
    simulate_intervention,
)
from core.thresholds import classify
from core.units import from_mgdl, to_mgdl

lang = get_current_language()
units = st.session_state.get("units", "mmol")

# Demo profiles (lab values in mg/dL, the model's native unit). HDL None = "not known".
PRESETS = {
    "preset_a": {"male": 0, "age": 35, "cigsPerDay": 0, "BPMeds": 0, "sysBP": 115, "diaBP": 75,
                 "height": 165, "weight": 58.0, "totChol": 170.0, "HDL": 60.0, "glucose": 85.0, "diabetes": 0},
    "preset_b": {"male": 1, "age": 52, "cigsPerDay": 20, "BPMeds": 0, "sysBP": 142, "diaBP": 88,
                 "height": 176, "weight": 84.0, "totChol": 225.0, "HDL": None, "glucose": 98.0, "diabetes": 0},
    "preset_c": {"male": 0, "age": 60, "cigsPerDay": 0, "BPMeds": 1, "sysBP": 155, "diaBP": 92,
                 "height": 160, "weight": 75.5, "totChol": 240.0, "HDL": 50.0, "glucose": 105.0, "diabetes": 0},
    "preset_d": {"male": 1, "age": 66, "cigsPerDay": 25, "BPMeds": 1, "sysBP": 168, "diaBP": 98,
                 "height": 174, "weight": 96.0, "totChol": 260.0, "HDL": 35.0, "glucose": 145.0, "diabetes": 1},
}


def with_derived(p: dict) -> dict:
    """Fill the model inputs that are derived rather than asked."""
    p = dict(p)
    p["currentSmoker"] = 1 if float(p.get("cigsPerDay", 0)) > 0 else 0
    p["BMI"] = round(float(p["weight"]) / ((float(p["height"]) / 100.0) ** 2), 1)
    p["prevalentHyp"] = 1 if (p["BPMeds"] or p["sysBP"] >= 140 or p["diaBP"] >= 90) else 0
    p["heartRate"] = FIXED_HEART_RATE
    p["prevalentStroke"] = 0
    p.setdefault("prior_cvd", 0)
    return p


@st.cache_data(max_entries=128, show_spinner=False)
def evaluate(items: tuple):
    p = dict(items)
    return predict_patient(p), generate_recommendations(p), explain_prediction(p, head="chd", m_steps=50)


@st.cache_data(max_entries=256, show_spinner=False)
def simulate(items: tuple, changes: tuple):
    return simulate_intervention(dict(items), dict(changes))


@st.cache_data(max_entries=64, show_spinner=False)
def trajectory(items: tuple, changes: tuple):
    return compute_risk_trajectory(
        dict(items), years_ahead=(0, 2, 5, 8, 10, 12, 15, 18, 20), improved_changes=dict(changes) or None
    )


@st.cache_data(max_entries=32, show_spinner=False)
def pdf_bytes(items: tuple, lang_: str, units_: str):
    p = dict(items)
    pred, recs, _ = evaluate(items)
    return generate_pdf_report(p, pred, recs, lang=lang_, units=units_)


def load_preset(key: str) -> None:
    st.session_state["patient_data"] = with_derived(PRESETS[key])
    st.session_state["form_ver"] += 1


if "patient_data" not in st.session_state or "height" not in st.session_state["patient_data"]:
    st.session_state["patient_data"] = with_derived(PRESETS["preset_b"])
st.session_state.setdefault("form_ver", 0)

cur = st.session_state["patient_data"]
ver = st.session_state["form_ver"]
uk = f"{ver}_{units}"
u_label = {"en": "mmol/L", "uz": "mmol/l"}.get(lang, "ммоль/л") if units == "mmol" else "mg/dL"


def lab_input(label_key: str, analyte: str, mg_value, mmol_range, mg_range, optional=False):
    """Number input in the selected unit; returns mg/dL (or None if left empty)."""
    label = f"{t(label_key, lang=lang)}, {u_label}"
    help_key = f"{label_key}_help"
    if units == "mmol":
        lo, hi, step, fmt = mmol_range[0], mmol_range[1], mmol_range[2], "%.2f" if mmol_range[2] < 0.1 else "%.1f"
        val = None if mg_value is None else round(from_mgdl(mg_value, analyte, "mmol"), 2)
    else:
        lo, hi, step, fmt = mg_range[0], mg_range[1], mg_range[2], "%.0f"
        val = None if mg_value is None else round(float(mg_value))
    if val is not None:
        val = float(min(hi, max(lo, val)))
    entered = st.number_input(
        label, min_value=float(lo), max_value=float(hi), value=val, step=float(step), format=fmt,
        placeholder=t("hdl_placeholder", lang=lang) if optional else None,
        help=t(help_key, lang=lang), key=f"{analyte}_{label_key}_{uk}",
    )
    if entered is None:
        return None
    return to_mgdl(entered, analyte, units)


# ---------------------------------------------------------------- header
st.markdown(logo_header_html(t("app_title", lang=lang), t("app_subtitle", lang=lang)), unsafe_allow_html=True)
st.caption(t("screen_disclaimer", lang=lang))

col_form, col_res = st.columns([5, 7], gap="large")

# ---------------------------------------------------------------- form
with col_form:
    st.markdown(f"**{t('btn_preset_profile', lang=lang)}**")
    with st.container(horizontal=True):
        for p_key in PRESETS:
            st.button(t(p_key, lang=lang), key=f"btn_{p_key}", on_click=load_preset, args=(p_key,),
                      help=t(f"{p_key}_help", lang=lang))

    with st.form("patient_form"):
        st.markdown(f"##### {t('section_demographics', lang=lang)}")
        sex = st.segmented_control(t("field_sex", lang=lang), ["m", "f"],
                                   format_func=lambda s: t("sex_male" if s == "m" else "sex_female", lang=lang),
                                   default="m" if cur["male"] else "f", key=f"sex_{uk}")
        c1, c3, c4 = st.columns(3)
        age = c1.number_input(t("field_age", lang=lang), MIN_TRAIN_AGE, MAX_TRAIN_AGE,
                              int(min(MAX_TRAIN_AGE, max(MIN_TRAIN_AGE, cur["age"]))), step=1,
                              help=t("field_age_help", lang=lang), key=f"age_{uk}")
        height = c3.number_input(t("calc_height_label", lang=lang), 120, 220, int(cur["height"]), step=1,
                                 key=f"height_{uk}")
        weight = c4.number_input(t("calc_weight_label", lang=lang), 35.0, 200.0, float(cur["weight"]), step=0.5,
                                 format="%.1f", help=t("field_weight_help", lang=lang), key=f"weight_{uk}")

        st.markdown(f"##### {t('section_vitals', lang=lang)}")
        c5, c6 = st.columns(2)
        sbp = c5.number_input(t("field_sys_bp", lang=lang), 80, 230, int(cur["sysBP"]), step=1,
                              help=t("field_sys_bp_help", lang=lang), key=f"sbp_{uk}")
        dbp = c6.number_input(t("field_dia_bp", lang=lang), 40, 140, int(cur["diaBP"]), step=1, key=f"dbp_{uk}")
        bp_meds = st.toggle(t("field_bp_meds", lang=lang), value=bool(cur["BPMeds"]),
                            help=t("field_bp_meds_help", lang=lang), key=f"meds_{uk}")

        st.markdown(f"##### {t('section_lab', lang=lang)}")
        tc = lab_input("field_tot_chol", "totChol", cur["totChol"], (2.5, 12.0, 0.1), (100, 450, 1))
        hdl = lab_input("field_hdl", "totChol", cur.get("HDL"), (0.5, 3.0, 0.05), (20, 120, 1), optional=True)
        glu = lab_input("field_glucose", "glucose", cur["glucose"], (3.0, 20.0, 0.1), (55, 360, 1))
        known_dm = st.toggle(t("field_known_diabetes", lang=lang), value=bool(cur["diabetes"]), key=f"dm_{uk}")

        st.markdown(f"##### {t('section_lifestyle', lang=lang)}")
        cigs = st.number_input(t("field_cigs_per_day", lang=lang), 0, 60, int(cur["cigsPerDay"]), step=1,
                               help=t("field_cigs_help", lang=lang), key=f"cigs_{uk}")
        prior_cvd = st.checkbox(t("field_prior_cvd", lang=lang), value=bool(cur.get("prior_cvd", 0)),
                                help=t("field_prior_cvd_help", lang=lang), key=f"cvd_{uk}")

        submitted = st.form_submit_button(t("btn_calculate", lang=lang), type="primary",
                                          icon=":material/monitor_heart:", width="stretch")

    if submitted:
        st.session_state["patient_data"] = with_derived({
            "male": 1 if (sex or ("m" if cur["male"] else "f")) == "m" else 0,
            "age": int(age), "height": int(height), "weight": float(weight),
            "sysBP": float(sbp), "diaBP": float(dbp), "BPMeds": int(bp_meds),
            "totChol": float(tc), "HDL": float(hdl) if hdl else None, "glucose": float(glu),
            "diabetes": int(known_dm), "cigsPerDay": float(cigs), "prior_cvd": int(prior_cvd),
        })
        cur = st.session_state["patient_data"]

# ---------------------------------------------------------------- results
patient = st.session_state["patient_data"]
items = tuple(sorted(patient.items()))

with col_res:
    if patient.get("prior_cvd"):
        with st.container(border=True):
            st.markdown(f"### :material/emergency: {t('prior_cvd_title', lang=lang)}")
            st.write(t("prior_cvd_body", lang=lang))
        st.stop()

    prediction, recs, xai = evaluate(items)
    age_v = float(patient["age"])
    heart_age = prediction["heart_age"]
    ha_delta = prediction["heart_age_delta"]
    p_chd = prediction["p_chd"] * 100.0
    p_dm = prediction["p_diabetes"] * 100.0
    life = calculate_life_expectancy_estimate(age_v, patient["male"], heart_age)

    if p_chd < 10.0:
        cat_label, cat_color = t("card_chd_cat_low", lang=lang), "green"
    elif p_chd < 20.0:
        cat_label, cat_color = t("card_chd_cat_mod", lang=lang), "orange"
    else:
        cat_label, cat_color = t("card_chd_cat_high", lang=lang), "red"

    if ha_delta >= 1.0:
        sub_key = "verdict_sub_older"
    elif ha_delta <= -1.0:
        sub_key = "verdict_sub_younger"
    else:
        sub_key = "verdict_sub_same"
    st.markdown(
        f'<div class="yurak-verdict">{t("verdict_head", lang=lang, ha=f"{heart_age:.0f}")}</div>'
        f'<div class="yurak-verdict-sub">{t(sub_key, lang=lang, age=int(age_v), diff=f"{abs(ha_delta):.1f}")}</div>',
        unsafe_allow_html=True,
    )

    with st.container(horizontal=True):
        with st.container(border=True):
            st.metric(t("res_label_chd", lang=lang), f"{p_chd:.1f}%", help=t("card_chd_help", lang=lang))
            st.badge(cat_label, color=cat_color)
        with st.container(border=True):
            st.metric(t("res_label_life_exp", lang=lang), f"{life['life_estimate']:.1f}",
                      help=t("card_life_exp_help", lang=lang))
            d = life["delta"]
            st.badge(t("life_vs_avg", lang=lang, val=f"{d:+.1f}"),
                     color="red" if d <= -1 else ("green" if d >= 1 else "gray"))
        with st.container(border=True):
            if patient["diabetes"]:
                st.metric(t("res_label_diabetes", lang=lang), t("dm_known_value", lang=lang),
                          help=t("card_dm_known_help", lang=lang))
                st.badge(t("dm_known_badge", lang=lang), color="gray")
            else:
                st.metric(t("res_label_diabetes", lang=lang), f"{p_dm:.1f}%", help=t("card_dm_help", lang=lang))
                high_dm = p_dm >= 50.0 or prediction["diabetes_flag"]
                st.badge(t("card_dm_high" if high_dm else "card_dm_low", lang=lang),
                         color="red" if high_dm else "green")

    grouped = group_attributions(xai["attributions_pp"])
    drivers = sorted(((k, v) for k, v in grouped.items() if k not in ("age", "male")), key=lambda kv: kv[1], reverse=True)
    if drivers and drivers[0][1] >= 0.5:
        st.markdown(t("driver_line", lang=lang, driver=t(f"feat_{drivers[0][0]}", lang=lang),
                      contrib=f"{drivers[0][1]:.1f}"))

    # ------------------------------------------------------------ steps
    st.subheader(t("steps_title", lang=lang))
    all_items = recs.get("all_interventions", [])
    selected_changes: dict = {}
    if not all_items:
        st.success(t("rec_maintenance_optimal", lang=lang), icon=":material/favorite:")
    else:
        st.caption(t("steps_sub", lang=lang))
        for item in all_items:
            with st.container(border=True):
                on = st.checkbox(
                    f"**{t(item['title_key'], lang=lang)}**", value=True, key=f"step_{item['id']}_{ver}",
                    help=f"{t(item['rationale_key'], lang=lang)}\n\n{t('rec_source', lang=lang)}: {item['source']}",
                )
                effects = []
                if abs(item["delta_chd_pp"]) >= 0.05:
                    effects.append(t("eff_chd", lang=lang, val=f"{abs(item['delta_chd_pp']):.1f}"))
                if abs(item["delta_heart_age"]) >= 0.1:
                    effects.append(t("eff_heart", lang=lang, val=f"{abs(item['delta_heart_age']):.1f}"))
                if abs(item["delta_life_exp"]) >= 0.1:
                    effects.append(t("eff_life", lang=lang, val=f"{abs(item['delta_life_exp']):.1f}"))
                if effects:
                    st.markdown(" · ".join(effects))
                st.caption(t(item["action_key"], lang=lang))
                if on:
                    selected_changes.update(item["changes"])

        if selected_changes:
            sim = simulate(items, tuple(sorted(selected_changes.items())))
            s, dd = sim["simulated"], sim["deltas"]
            st.markdown(f"**{t('steps_result_title', lang=lang)}**")
            with st.container(horizontal=True):
                st.metric(t("res_label_chd", lang=lang), f"{s['p_chd'] * 100:.1f}%",
                          delta=f"{dd['delta_chd_pp']:.1f} {t('sim_pp', lang=lang)}", delta_color="inverse", border=True)
                st.metric(t("res_label_heart_age", lang=lang), f"{s['heart_age']:.1f}",
                          delta=f"{dd['delta_heart_age']:.1f}", delta_color="inverse", border=True)
                st.metric(t("res_label_life_exp", lang=lang), f"{s['life_expectancy']:.1f}",
                          delta=f"{dd['delta_life_exp']:+.1f}", border=True)
        else:
            st.info(t("steps_none_selected", lang=lang), icon=":material/touch_app:")

# ---------------------------------------------------------------- details (full width)
st.divider()
tab_xai, tab_bio, tab_traj, tab_exp = st.tabs([
    f":material/bar_chart: {t('tab_xai', lang=lang)}",
    f":material/lab_profile: {t('tab_biomarkers', lang=lang)}",
    f":material/timeline: {t('tab_trajectory', lang=lang)}",
    f":material/description: {t('tab_export', lang=lang)}",
])

with tab_xai:
    st.caption(t("xai_desc", lang=lang))
    lbl_inc, lbl_dec = t("xai_type_increase", lang=lang), t("xai_type_decrease", lang=lang)
    rows = [
        {"feature": t(f"feat_{k}", lang=lang), "contribution": v, "type": lbl_inc if v > 0 else lbl_dec}
        for k, v in grouped.items()
        if abs(v) >= 0.05 and k not in ("age", "male", "heartRate", "prevalentStroke")
    ]
    if rows:
        df_xai = pd.DataFrame(rows).sort_values("contribution")
        chart = (
            alt.Chart(df_xai)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("contribution:Q", title=t("xai_axis_title", lang=lang)),
                y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("type:N", scale=alt.Scale(domain=[lbl_inc, lbl_dec], range=[BRAND_ANOR, BRAND_TEAL]),
                                title=None, legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("contribution:Q", format=".1f")],
            )
            .properties(height=max(160, 44 * len(rows)))
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info(t("xai_optimal_notice", lang=lang))

with tab_bio:
    st.caption(t("biomarkers_desc", lang=lang))
    sex_s = str(patient["male"])
    status = {"ok": t("status_badge_ok", lang=lang), "warn": t("status_badge_warn", lang=lang),
              "high": t("status_badge_high", lang=lang)}
    bp_unit = {"en": "mmHg", "uz": "mm sim.ust."}.get(lang, "мм рт. ст.")

    def lab_txt(mg, analyte):
        return f"{from_mgdl(mg, analyte, units):.2f} {u_label}" if units == "mmol" else f"{mg:.0f} mg/dL"

    bio = []
    c = classify("bp", patient["sysBP"], sex=sex_s, secondary_value=patient["diaBP"], lang=lang)
    bio.append((t("bio_marker_bp", lang=lang), f"{patient['sysBP']:.0f} / {patient['diaBP']:.0f} {bp_unit}", c))
    c = classify("totChol", patient["totChol"], sex=sex_s, lang=lang)
    bio.append((t("field_tot_chol", lang=lang), lab_txt(patient["totChol"], "totChol"), c))
    if patient.get("HDL"):
        c = classify("hdl", patient["HDL"], sex=sex_s, lang=lang)
        bio.append((t("field_hdl", lang=lang), lab_txt(patient["HDL"], "totChol"), c))
    c = classify("glucose", patient["glucose"], sex=sex_s, lang=lang)
    bio.append((t("field_glucose", lang=lang), lab_txt(patient["glucose"], "glucose"), c))
    c = classify("bmi", patient["BMI"], sex=sex_s, lang=lang)
    c = {**c, "normal_range": "18.5–24.9 kg/m²"}
    bio.append((t("bio_marker_bmi", lang=lang), f"{patient['BMI']:.1f} kg/m²", c))
    smoke_c = {"level": "high" if patient["currentSmoker"] else "ok", "normal_range": t("bio_cessation_norm", lang=lang),
               "source": "WHO FCTC / ESC 2021"}
    smoke_v = (t("bio_smoker_val", lang=lang, cigs=int(patient["cigsPerDay"])) if patient["currentSmoker"]
               else t("bio_nonsmoker_val", lang=lang))
    bio.append((t("bio_marker_smoker", lang=lang), smoke_v, smoke_c))

    df_bio = pd.DataFrame([
        {t("bio_col_name", lang=lang): n, t("bio_col_val", lang=lang): v,
         t("bio_col_norm", lang=lang): c.get("normal_range", ""), t("bio_col_status", lang=lang): status[c["level"]],
         t("bio_col_source", lang=lang): c.get("source", "")}
        for n, v, c in bio
    ])
    st.dataframe(df_bio, hide_index=True, width="stretch")
    if not patient.get("HDL"):
        st.caption(t("hdl_default_note", lang=lang))

with tab_traj:
    st.caption(t("traj_desc", lang=lang))
    traj = pd.DataFrame(trajectory(items, tuple(sorted(selected_changes.items()))))
    traj["scenario_label"] = traj["scenario"].map({
        "current": t("traj_scenario_current", lang=lang),
        "improved": t("traj_scenario_improved", lang=lang),
    })
    max_age = float(traj["age"].max())
    layers = []
    if max_age > MAX_TRAIN_AGE:
        band = pd.DataFrame({"x0": [float(MAX_TRAIN_AGE)], "x1": [max_age],
                             "label": [t("traj_extrapolated", lang=lang)]})
        layers.append(alt.Chart(band).mark_rect(color="#152238", opacity=0.06)
                      .encode(x="x0:Q", x2="x1:Q", tooltip=["label:N"]))
    layers.append(
        alt.Chart(traj).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("age:Q", title=t("traj_axis_age", lang=lang), scale=alt.Scale(zero=False)),
            y=alt.Y("p_chd_percent:Q", title=t("traj_axis_risk", lang=lang)),
            color=alt.Color("scenario_label:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=[t("traj_scenario_current", lang=lang),
                                                    t("traj_scenario_improved", lang=lang)],
                                            range=[BRAND_ANOR, BRAND_TEAL])),
            tooltip=[alt.Tooltip("age:Q"), alt.Tooltip("p_chd_percent:Q", format=".1f"), "scenario_label:N"],
        )
    )
    st.altair_chart(alt.layer(*layers).properties(height=340), width="stretch")
    if max_age > MAX_TRAIN_AGE:
        st.caption(t("traj_extrapolated_note", lang=lang))

with tab_exp:
    c_pdf, c_fhir = st.columns(2)
    with c_pdf:
        with st.container(border=True):
            st.markdown(f"**{t('export_report_title', lang=lang)}**")
            st.caption(t("export_report_desc", lang=lang))
            st.download_button(
                t("export_pdf_btn", lang=lang), data=pdf_bytes(items, lang, units),
                file_name=f"Yurak_{int(patient['age'])}_{'M' if patient['male'] else 'F'}.pdf",
                mime="application/pdf", type="primary", icon=":material/download:", width="stretch",
            )
    with c_fhir:
        with st.container(border=True):
            st.markdown(f"**{t('export_fhir_title', lang=lang)}**")
            st.caption(t("export_fhir_desc", lang=lang))
            st.download_button(
                t("export_fhir_btn", lang=lang),
                data=json.dumps(generate_fhir_bundle(patient, prediction, lang=lang), indent=2, ensure_ascii=False),
                file_name="yurak_fhir_bundle.json", mime="application/json",
                icon=":material/data_object:", width="stretch",
            )
