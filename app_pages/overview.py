"""
app_pages/overview.py — Landing page: what you will learn, why you can trust it, and the way in.
"""
from pathlib import Path

import streamlit as st

from core.branding import PROMO_VIDEO_PATH
from core.i18n import get_current_language, t

lang = get_current_language()

col_text, col_video = st.columns([5, 7], gap="large")

with col_text:
    st.markdown(f"# {t('ov_head', lang=lang)}")
    st.write(t("ov_lead", lang=lang))
    if st.button(t("ov_cta", lang=lang), type="primary", icon=":material/monitor_heart:", key="ov_cta_btn"):
        st.switch_page("app_pages/screening.py")
    st.caption(t("ov_cta_hint", lang=lang))
    with st.container(horizontal=True):
        for key in ("ov_trust_free", "ov_trust_private", "ov_trust_lang"):
            st.badge(t(key, lang=lang), color="gray")

with col_video:
    if Path(PROMO_VIDEO_PATH).is_file():
        st.video(str(PROMO_VIDEO_PATH))
        st.caption(t("ov_video_note", lang=lang))
    else:
        st.info(t("ov_video_missing", lang=lang), icon=":material/videocam_off:")

st.write("")
st.markdown(f"### {t('ov_what_title', lang=lang)}")
cols = st.columns(4)
for col, (icon, key) in zip(cols, ((":material/cardiology:", "ov_f1"), (":material/monitoring:", "ov_f2"),
                                   (":material/checklist:", "ov_f3"), (":material/description:", "ov_f4"))):
    with col:
        with st.container(border=True, height="stretch"):
            st.markdown(f"#### {icon} {t(key + '_t', lang=lang)}")
            st.write(t(key + "_d", lang=lang))

st.write("")
st.markdown(f"### {t('ov_need_title', lang=lang)}")
st.write(t("ov_need_body", lang=lang))

st.write("")
st.caption(t("ov_author", lang=lang))
st.caption(t("ov_disclaimer", lang=lang))
