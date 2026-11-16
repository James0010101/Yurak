"""
core/branding.py — Logo and shared visual identity of Yurak.

Palette (also mirrored in .streamlit/config.toml):
- Tile teal  #0E6E6B  — primary actions, links, "good" direction. Taken from
  the turquoise majolica of Samarkand, calm and clinical without being blue-hospital.
- Lapis ink  #152238  — text.
- Anor red   #A8203A  — pomegranate: the logo and high-risk states only.
- Mineral    #F3F6F5 / #E5ECEA — backgrounds.
"""
import base64
from functools import lru_cache
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_SVG_PATH = ASSETS_DIR / "logo.svg"
LOGO_PNG_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"
PROMO_VIDEO_PATH = ASSETS_DIR / "promo_uz.mp4"

BRAND_TEAL = "#0E6E6B"
BRAND_INK = "#152238"
BRAND_ANOR = "#A8203A"
RISK_LOW = "#2E7D4F"
RISK_MID = "#B5650D"
RISK_HIGH = BRAND_ANOR


def risk_color(p_chd_percent: float) -> str:
    """Category colour for a 10-year CHD risk in percent (NCEP ATP III bands)."""
    if p_chd_percent < 10.0:
        return RISK_LOW
    if p_chd_percent < 20.0:
        return RISK_MID
    return RISK_HIGH


@lru_cache(maxsize=1)
def logo_data_uri() -> str:
    """Logo as an inline data URI, usable inside HTML markup."""
    encoded = base64.b64encode(LOGO_SVG_PATH.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def logo_header_html(title: str, subtitle: str = "", size: int = 44) -> str:
    """Logo + wordmark row used at the top of pages."""
    sub = f'<div class="yurak-sub">{subtitle}</div>' if subtitle else ""
    return (
        '<div class="yurak-brand">'
        f'<img src="{logo_data_uri()}" width="{size}" height="{size}" alt=""/>'
        f'<div><div class="yurak-word">{title}</div>{sub}</div>'
        "</div>"
    )


# The only custom CSS in the app: the wordmark row and the result headline.
# Everything else (colours, fonts, radii) comes from .streamlit/config.toml.
BRAND_CSS = f"""
<style>
.yurak-brand {{ display:flex; align-items:center; gap:14px; margin:0 0 2px 0; }}
.yurak-word {{ font-size:2rem; font-weight:800; letter-spacing:-0.03em; line-height:1; color:{BRAND_INK}; }}
.yurak-sub {{ font-size:.95rem; color:#4A5A70; margin-top:4px; }}
.yurak-verdict {{ font-size:2.6rem; font-weight:800; letter-spacing:-0.03em; line-height:1.05; color:{BRAND_INK}; margin:0; }}
.yurak-verdict-sub {{ font-size:1.05rem; color:#3E4D63; margin:10px 0 0 0; line-height:1.45; max-width:62ch; }}
/* Result headline: a tinted panel whose left edge carries the risk band colour. */
[class*="st-key-verdict_"] {{ padding:22px 24px 18px; border-radius:14px; background:#FFFFFF;
  border:1px solid #CFDAD7; border-left-width:6px; }}
.st-key-verdict_low {{ border-left-color:{RISK_LOW}; }}
.st-key-verdict_mid {{ border-left-color:{RISK_MID}; }}
.st-key-verdict_high {{ border-left-color:{RISK_HIGH}; }}
@media (max-width: 640px) {{ .yurak-verdict {{ font-size:1.9rem; }} .yurak-word {{ font-size:1.5rem; }} }}
</style>
"""
