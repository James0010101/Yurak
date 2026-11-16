"""
streamlit_app.py — Main entry point for Yurak.

No sidebar: a phone-first header (wordmark + one settings popover) sits above
every page, navigation is the top bar.
"""
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from core.i18n import SUPPORTED_LANGUAGES, get_current_language, set_current_language, t
from core.branding import BRAND_CSS, FAVICON_PATH, logo_header_html

st.set_page_config(
    page_title="Yurak",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(BRAND_CSS, unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state["lang"] = "ru"
if "units" not in st.session_state:
    st.session_state["units"] = "mmol"

pages = [
    st.Page("app_pages/overview.py", title=t("nav_overview"), icon=":material/home:", default=True),
    st.Page("app_pages/screening.py", title=t("nav_screening"), icon=":material/cardiology:"),
    st.Page("app_pages/methodology.py", title=t("nav_methodology"), icon=":material/science:"),
    st.Page("app_pages/about.py", title=t("nav_about"), icon=":material/info:"),
]
nav = st.navigation(pages, position="top")

lang_labels = {"ru": "Русский", "uz": "Oʻzbekcha", "en": "English"}
head_l, head_r = st.columns([4, 1], vertical_alignment="center")
with head_l:
    st.markdown(logo_header_html("Yurak", t("app_subtitle"), size=40), unsafe_allow_html=True)
with head_r:
    current_lang = get_current_language()
    with st.popover(f"{lang_labels.get(current_lang, current_lang)} · {t('unit_' + st.session_state['units'])}",
                    icon=":material/tune:", width="stretch"):
        selected_lang = st.selectbox(
            t("lang_select"),
            options=SUPPORTED_LANGUAGES,
            format_func=lambda x: lang_labels.get(x, x),
            index=SUPPORTED_LANGUAGES.index(current_lang) if current_lang in SUPPORTED_LANGUAGES else 0,
            key="lang_selector",
        )
        unit_labels = {"mmol": t("unit_mmol"), "mgdl": t("unit_mgdl")}
        selected_unit = st.segmented_control(
            t("unit_select"), options=["mmol", "mgdl"], format_func=lambda x: unit_labels.get(x, x),
            default=st.session_state["units"], key="unit_selector",
        )
        st.caption(t("unit_hint"))
    if selected_lang != st.session_state["lang"]:
        set_current_language(selected_lang)
        st.rerun()
    if selected_unit and selected_unit != st.session_state["units"]:
        st.session_state["units"] = selected_unit
        st.rerun()

nav.run()

st.divider()
st.caption(t("footer_disclaimer"))
