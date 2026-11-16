"""
app_pages/overview.py — Landing page: what Yurak tells you, the promo video, and the way in.
"""
from pathlib import Path

import streamlit as st

from core.branding import PROMO_VIDEO_PATH, logo_header_html
from core.i18n import get_current_language, t

lang = get_current_language()

st.markdown(logo_header_html(t("app_title", lang=lang), t("ov_tagline", lang=lang), size=56),
            unsafe_allow_html=True)
st.write("")

col_text, col_video = st.columns([5, 7], gap="large")

with col_text:
    st.markdown(f"## {t('ov_head', lang=lang)}")
    st.write(t("ov_lead", lang=lang))
    for icon, key in ((":material/cardiology:", "ov_f1"), (":material/insights:", "ov_f2"),
                      (":material/checklist:", "ov_f3"), (":material/description:", "ov_f4")):
        st.markdown(f"{icon} **{t(key + '_t', lang=lang)}.** {t(key + '_d', lang=lang)}")
    st.write("")
    st.page_link("app_pages/screening.py", label=t("ov_cta", lang=lang), icon=":material/arrow_forward:")
    st.caption(t("ov_cta_hint", lang=lang))

with col_video:
    if Path(PROMO_VIDEO_PATH).is_file():
        st.video(str(PROMO_VIDEO_PATH))
        st.caption(t("ov_video_note", lang=lang))
    else:
        st.info(t("ov_video_missing", lang=lang), icon=":material/videocam_off:")

st.divider()
st.caption(t("ov_author", lang=lang))
st.caption(t("ov_disclaimer", lang=lang))
