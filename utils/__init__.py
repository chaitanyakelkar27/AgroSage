"""
AgroSage — Utils Package
Centralised re-exports for shared helpers used across all Streamlit pages.
"""

import os
import streamlit as st


# ─── CSS Loader ───────────────────────────────────────────────────────────────

def load_css(filepath: str = None) -> None:
    """
    Inject a CSS file into the active Streamlit page.

    Parameters
    ----------
    filepath : str, optional
        Absolute or relative path to the CSS file.
        Defaults to ``assets/style.css`` relative to the project root.

    If the file does not exist, a silent warning is printed to the
    terminal (not shown to the user) and no CSS is injected.
    """
    if filepath is None:
        # Resolve relative to project root (one level above utils/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(project_root, "assets", "style.css")

    if not os.path.isfile(filepath):
        print(f"[AgroSage] CSS file not found: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        css = f.read()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render a consistent cross-page sidebar navigation."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
                <div style="
                    width: 42px; height: 42px;
                    background: linear-gradient(135deg, #40916C, #52B788);
                    border-radius: 10px;
                    display: inline-flex; align-items: center; justify-content: center;
                    margin-bottom: 0.5rem;
                ">
                    <span style="font-size: 1.1rem; font-weight: 800; color: #fff;
                                 font-family: var(--font-body); letter-spacing: -0.04em;">Ag</span>
                </div>
                <h2 style="margin: 0; font-size: 1.15rem; font-weight: 700;
                           letter-spacing: -0.02em; color: #FFFFFF !important;">
                    AgroSage
                </h2>
                <p style="margin: 0.15rem 0 0 0; font-size: 0.68rem; font-weight: 500;
                          opacity: 0.5; letter-spacing: 0.06em; text-transform: uppercase;">
                    Crop Advisory Platform
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            "<p style='font-size:0.66rem; font-weight:600; text-transform:uppercase; "
            "letter-spacing:0.08em; opacity:0.4; margin-bottom:0.4rem;'>Navigation</p>",
            unsafe_allow_html=True,
        )

        st.page_link("Home.py", label="Dashboard", use_container_width=True)
        st.page_link("pages/1_Crop_Recommender.py", label="Crop Recommender", use_container_width=True)
        st.page_link("pages/2_Disease_Detector.py", label="Disease Detector", use_container_width=True)
        st.page_link("pages/3_Weather_Analyst.py", label="Weather Analyst", use_container_width=True)

        st.markdown("---")

        st.markdown(
            """
            <div style="text-align:center; padding:0.25rem 0;">
                <p style="font-size:0.65rem; font-weight:400; opacity:0.35;
                          font-family: var(--font-mono); margin:0;">
                    v1.0.0
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── Re-exports ──────────────────────────────────────────────────────────────
# Import after definition so circular issues are avoided.

from utils.weather_utils import (          # noqa: E402, F401
    load_climate_csv,
    get_monthly_stats,
    get_mock_weather,
)

from utils.crop_predictor import (         # noqa: E402, F401
    load_crop_models,
    predict_crop,
)

from utils.disease_predictor import (      # noqa: E402, F401
    load_disease_models,
    preprocess_image,
    clean_image,
    parse_disease_label,
    predict_disease,
)

__all__ = [
    "load_css",
    "render_sidebar",
    "load_climate_csv",
    "get_monthly_stats",
    "get_mock_weather",
    "load_crop_models",
    "predict_crop",
    "load_disease_models",
    "preprocess_image",
    "clean_image",
    "parse_disease_label",
    "predict_disease",
]

