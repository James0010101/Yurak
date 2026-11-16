"""
streamlit_app.py — Main entry point for Yurak.
"""
from pathlib import Path
import sys

# Ensure root workspace path is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from core.i18n import SUPPORTED_LANGUAGES, get_current_language, set_current_language, t
from core.branding import BRAND_CSS, FAVICON_PATH, LOGO_PNG_PATH

st.set_page_config(
    page_title="Yurak",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(BRAND_CSS, unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state["lang"] = "ru"
if "units" not in st.session_state:
    st.session_state["units"] = "mmol"

with st.sidebar:
    st.logo(str(LOGO_PNG_PATH), size="large")
    st.markdown("### Yurak")
    st.caption(t("app_subtitle"))

    lang_labels = {"ru": "Русский", "uz": "Oʻzbekcha", "en": "English"}
    current_lang = get_current_language()
    selected_lang = st.selectbox(
        t("lang_select"),
        options=SUPPORTED_LANGUAGES,
        format_func=lambda x: lang_labels.get(x, x),
        index=SUPPORTED_LANGUAGES.index(current_lang) if current_lang in SUPPORTED_LANGUAGES else 0,
        key="lang_selector",
    )
    if selected_lang != st.session_state["lang"]:
        set_current_language(selected_lang)
        st.rerun()

    unit_labels = {"mmol": t("unit_mmol"), "mgdl": t("unit_mgdl")}
    selected_unit = st.segmented_control(
        t("unit_select"),
        options=["mmol", "mgdl"],
        format_func=lambda x: unit_labels.get(x, x),
        default=st.session_state["units"],
        key="unit_selector",
    )
    if selected_unit:
        st.session_state["units"] = selected_unit

    st.divider()
    st.caption(t("footer_disclaimer"))

pages = [
    st.Page("app_pages/overview.py", title=t("nav_overview"), icon=":material/play_circle:", default=True),
    st.Page("app_pages/screening.py", title=t("nav_screening"), icon=":material/cardiology:"),
    st.Page("app_pages/methodology.py", title=t("nav_methodology"), icon=":material/science:"),
    st.Page("app_pages/about.py", title=t("nav_about"), icon=":material/info:"),
]

nav = st.navigation(pages, position="top")
nav.run()
