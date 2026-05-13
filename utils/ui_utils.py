"""
AgroSage — UI Utility Functions
=================================
Helper functions for the icon-based visual UI designed for farmers
who cannot read text. Provides:

    render_icon_nav()         → Large emoji navigation cards
    render_severity_badge()   → Disease severity visual indicator
    render_step_guide()       → Illustrated step-by-step guide
    render_voice_button()     → Large microphone button
    render_result_with_audio()→ Result display with auto-generated audio
"""

from __future__ import annotations

import streamlit as st


# ── Icon Navigation Grid ─────────────────────────────────────────────────────

_NAV_ITEMS = [
    {
        "emoji": "🌱",
        "label": "Crop Recommender",
        "page": "pages/1_Crop_Recommender.py",
        "color_start": "#2D6A4F",
        "color_end": "#40916C",
        "desc": "Find the best crop",
    },
    {
        "emoji": "🍃",
        "label": "Disease Detector",
        "page": "pages/2_Disease_Detector.py",
        "color_start": "#1B6B93",
        "color_end": "#4FC0D0",
        "desc": "Check leaf health",
    },
    {
        "emoji": "🌤️",
        "label": "Weather",
        "page": "pages/3_Weather_Analyst.py",
        "color_start": "#6C3483",
        "color_end": "#A569BD",
        "desc": "See weather data",
    },
    {
        "emoji": "💬",
        "label": "Ask a Question",
        "page": "pages/4_Agronomy_Assistant.py",
        "color_start": "#6C5B2A",
        "color_end": "#C8A24A",
        "desc": "Talk to assistant",
    },
]


def render_icon_nav() -> None:
    """
    Render a grid of large emoji-based navigation cards.
    Each card links to a specific page. Designed for zero-text navigation
    so illiterate farmers can navigate visually.
    """
    cols = st.columns(4)

    for col, item in zip(cols, _NAV_ITEMS):
        with col:
            st.markdown(
                f"""
                <div class="ag-icon-card" style="
                    background: linear-gradient(135deg, {item['color_start']} 0%, {item['color_end']} 100%);
                ">
                    <div class="ag-icon-emoji">{item['emoji']}</div>
                    <div class="ag-icon-label">{item['label']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.page_link(item["page"], label=f"{item['emoji']} {item['label']}", use_container_width=True)


# ── Severity Badge ───────────────────────────────────────────────────────────

_SEVERITY_CONFIG = {
    "healthy": {"emoji": "🟢", "label": "Healthy",  "color": "#27AE60", "bg": "rgba(39,174,96,0.12)"},
    "none":    {"emoji": "🟢", "label": "Healthy",  "color": "#27AE60", "bg": "rgba(39,174,96,0.12)"},
    "low":     {"emoji": "🟡", "label": "Mild",     "color": "#F39C12", "bg": "rgba(243,156,18,0.12)"},
    "medium":  {"emoji": "🟠", "label": "Moderate", "color": "#E67E22", "bg": "rgba(230,126,34,0.12)"},
    "high":    {"emoji": "🔴", "label": "Severe",   "color": "#E74C3C", "bg": "rgba(231,76,60,0.12)"},
}


def render_severity_badge(level: str) -> str:
    """
    Return HTML for a large, visual severity badge.

    Parameters
    ----------
    level : str
        One of: 'healthy', 'none', 'low', 'medium', 'high'

    Returns
    -------
    str
        HTML string for the severity badge.
    """
    config = _SEVERITY_CONFIG.get(level.lower(), _SEVERITY_CONFIG["low"])
    return (
        f'<div class="ag-severity-badge" style="background:{config["bg"]}; color:{config["color"]};">'
        f'  <span style="font-size:2rem; line-height:1;">{config["emoji"]}</span>'
        f'  <span style="font-size:1.1rem; font-weight:700; margin-left:0.5rem;">{config["label"]}</span>'
        f'</div>'
    )


def render_severity_inline(level: str) -> str:
    """
    Return a compact inline severity indicator for use in text.

    Parameters
    ----------
    level : str
        One of: 'healthy', 'none', 'low', 'medium', 'high'

    Returns
    -------
    str
        Short emoji + label string.
    """
    config = _SEVERITY_CONFIG.get(level.lower(), _SEVERITY_CONFIG["low"])
    return f"{config['emoji']} {config['label']}"


# ── Step-by-Step Guide ───────────────────────────────────────────────────────

def render_step_guide(steps: list[dict]) -> None:
    """
    Render an illustrated step-by-step guide using icons.

    Each step is a dict with:
        - emoji: str    (e.g., '📷')
        - label: str    (short text, optional — can be empty for zero-text)

    Parameters
    ----------
    steps : list[dict]
        List of step dictionaries.

    Example
    -------
    >>> render_step_guide([
    ...     {"emoji": "📷", "label": "Take Photo"},
    ...     {"emoji": "🔬", "label": "Analyze"},
    ...     {"emoji": "📋", "label": "Result"},
    ... ])
    """
    if not steps:
        return

    # Build the step guide HTML
    steps_html = ""
    for i, step in enumerate(steps):
        arrow = ' <span class="ag-step-arrow">→</span> ' if i < len(steps) - 1 else ""
        steps_html += (
            f'<div class="ag-step-item">'
            f'  <span class="ag-step-emoji">{step.get("emoji", "▶")}</span>'
            f'  <span class="ag-step-label">{step.get("label", "")}</span>'
            f'</div>'
            f'{arrow}'
        )

    st.markdown(
        f'<div class="ag-step-guide">{steps_html}</div>',
        unsafe_allow_html=True,
    )


# ── Language Selector ────────────────────────────────────────────────────────

_LANGUAGE_FLAGS = {
    "English": "🇬🇧",
    "Hindi": "🇮🇳",
    "Marathi": "🇮🇳",
    "Punjabi": "🇮🇳",
    "Tamil": "🇮🇳",
    "Telugu": "🇮🇳",
    "Kannada": "🇮🇳",
}


def render_language_selector(key: str = "ag_language") -> str:
    """
    Render a language selector dropdown and return the selected language name.

    Parameters
    ----------
    key : str
        Streamlit widget key for the selectbox.

    Returns
    -------
    str
        Selected language display name (e.g., 'Hindi', 'Marathi').
    """
    try:
        from utils.voice_utils import get_language_names
        languages = get_language_names()
    except ImportError:
        languages = ["English", "Hindi", "Marathi", "Punjabi", "Tamil", "Telugu", "Kannada"]

    # Format options with flags
    options = languages
    format_fn = lambda lang: f"{_LANGUAGE_FLAGS.get(lang, '🌐')} {lang}"

    selected = st.selectbox(
        "🗣️ Language / भाषा",
        options=options,
        format_func=format_fn,
        key=key,
        help="Select your language. Voice input and output will use this language.",
    )

    return selected


# ── Voice Result Display ─────────────────────────────────────────────────────

def render_result_with_audio(
    result_text: str,
    language_name: str = "English",
    auto_play: bool = True,
) -> None:
    """
    Render a result with auto-generated audio output.
    Shows the text and an audio player for the spoken version.

    Parameters
    ----------
    result_text : str
        The result text (in English — will be translated).
    language_name : str
        Target language for audio output.
    auto_play : bool
        Whether to automatically show the audio player.
    """
    if not result_text:
        return

    try:
        from utils.voice_utils import generate_result_audio
        audio_bytes = generate_result_audio(result_text, language_name)
        if audio_bytes and auto_play:
            st.audio(audio_bytes, format="audio/mp3")
    except ImportError:
        pass  # Voice utils not available, skip audio
    except Exception:
        pass  # Silently skip audio on any error


# ── Offline Banner ───────────────────────────────────────────────────────────

def render_offline_banner() -> None:
    """Show an offline mode warning banner if the app is offline."""
    try:
        from utils.offline_utils import is_offline_mode
        if is_offline_mode():
            st.markdown(
                """
                <div class="ag-offline-banner">
                    <span style="font-size:1.5rem; margin-right:0.5rem;">📡</span>
                    <span>Offline Mode — Weather data may be outdated. Crop and disease features work fully.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except ImportError:
        pass
