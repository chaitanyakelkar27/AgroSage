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


# ─── Re-exports ──────────────────────────────────────────────────────────────
# Import after definition so circular issues are avoided.

from utils.weather_utils import (          # noqa: E402, F401
    load_climate_csv,
    get_monthly_stats,
    get_mock_weather,
)

__all__ = [
    "load_css",
    "load_climate_csv",
    "get_monthly_stats",
    "get_mock_weather",
]
