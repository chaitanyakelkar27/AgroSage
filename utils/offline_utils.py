"""
AgroSage — Offline Support Utilities
======================================
Provides connectivity checking, weather data caching, and model path
helpers to ensure core features work without internet.

    check_connectivity()     → bool
    is_offline_mode()        → bool (with session state caching)
    load_weather_cache()     → dict or None
    save_weather_cache()     → None
    get_whisper_model_path() → str
"""

from __future__ import annotations

import json
import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
_DEFAULT_WEATHER_CACHE = os.path.join(_DATA_DIR, "weather_cache.json")

# Cache expiry: 7 days in seconds
_CACHE_EXPIRY_SECONDS = 7 * 24 * 60 * 60


# ── Connectivity Check ───────────────────────────────────────────────────────

def check_connectivity(timeout: float = 2.0) -> bool:
    """
    Check if the device has internet connectivity.

    Parameters
    ----------
    timeout : float
        Maximum seconds to wait for a response. Default is 2.0.

    Returns
    -------
    bool
        True if internet is reachable, False otherwise.
    """
    try:
        import requests
        response = requests.get(
            "https://www.google.com",
            timeout=timeout,
            allow_redirects=True,
        )
        return response.status_code == 200
    except Exception:
        return False


def is_offline_mode() -> bool:
    """
    Check connectivity and cache the result in Streamlit session state.
    Returns True if the device is offline.

    Uses session state key 'ag_offline_mode' to avoid repeated checks
    within the same Streamlit session.

    Returns
    -------
    bool
        True if offline, False if online.
    """
    try:
        import streamlit as st
        # Check once per session, re-check on explicit request
        if "ag_offline_mode" not in st.session_state:
            st.session_state["ag_offline_mode"] = not check_connectivity()
        return st.session_state["ag_offline_mode"]
    except ImportError:
        return not check_connectivity()


def refresh_connectivity() -> bool:
    """
    Force re-check connectivity and update session state.

    Returns
    -------
    bool
        True if offline, False if online.
    """
    try:
        import streamlit as st
        is_offline = not check_connectivity()
        st.session_state["ag_offline_mode"] = is_offline
        return is_offline
    except ImportError:
        return not check_connectivity()


# ── Weather Cache ────────────────────────────────────────────────────────────

def load_weather_cache(
    cache_path: Optional[str] = None,
    max_age_seconds: int = _CACHE_EXPIRY_SECONDS,
) -> Optional[dict]:
    """
    Load cached weather data from a JSON file.

    Parameters
    ----------
    cache_path : str, optional
        Path to the cache JSON file. Defaults to data/weather_cache.json.
    max_age_seconds : int
        Maximum age of cache in seconds before it's considered stale.
        Default is 7 days (604800 seconds).

    Returns
    -------
    dict or None
        Cached weather data dict if valid and fresh, None otherwise.
        The dict contains the weather data plus a '_cached_at' timestamp.
    """
    path = cache_path or _DEFAULT_WEATHER_CACHE

    if not os.path.isfile(path):
        logger.info(f"Weather cache not found: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check freshness
        cached_at = data.get("_cached_at", 0)
        age = time.time() - cached_at

        if age > max_age_seconds:
            logger.info(f"Weather cache expired ({age:.0f}s old, max {max_age_seconds}s)")
            return None

        logger.info(f"Weather cache loaded ({age:.0f}s old)")
        return data

    except (json.JSONDecodeError, OSError) as exc:
        logger.error(f"Failed to load weather cache: {exc}")
        return None


def save_weather_cache(
    data: dict,
    cache_path: Optional[str] = None,
) -> None:
    """
    Save weather data to a local JSON cache file with a timestamp.

    Parameters
    ----------
    data : dict
        Weather data to cache. A '_cached_at' timestamp will be added.
    cache_path : str, optional
        Path to the cache JSON file. Defaults to data/weather_cache.json.
    """
    path = cache_path or _DEFAULT_WEATHER_CACHE

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        cache_data = {**data, "_cached_at": time.time()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, default=str)

        logger.info(f"Weather cache saved to {path}")

    except OSError as exc:
        logger.error(f"Failed to save weather cache: {exc}")


# ── Whisper Model Path ───────────────────────────────────────────────────────

def get_whisper_model_path() -> str:
    """
    Return the local path for the cached Whisper tiny model.
    Creates the directory if it doesn't exist.

    Returns
    -------
    str
        Absolute path to the Whisper model cache directory.
    """
    path = os.path.join(_MODEL_DIR, "whisper-tiny")
    os.makedirs(path, exist_ok=True)
    return path


# ── ONNX / TFLite Model Paths ───────────────────────────────────────────────

def get_onnx_crop_model_path() -> Optional[str]:
    """
    Return path to crop_model.onnx if it exists.

    Returns
    -------
    str or None
        Path if the ONNX model exists, None otherwise.
    """
    path = os.path.join(_MODEL_DIR, "crop_model.onnx")
    return path if os.path.isfile(path) else None


def get_tflite_disease_model_path() -> Optional[str]:
    """
    Return path to disease_cnn.tflite if it exists.

    Returns
    -------
    str or None
        Path if the TFLite model exists, None otherwise.
    """
    path = os.path.join(_MODEL_DIR, "disease_cnn.tflite")
    return path if os.path.isfile(path) else None


def get_tflite_autoencoder_path() -> Optional[str]:
    """
    Return path to autoencoder.tflite if it exists.

    Returns
    -------
    str or None
        Path if the TFLite model exists, None otherwise.
    """
    path = os.path.join(_MODEL_DIR, "autoencoder.tflite")
    return path if os.path.isfile(path) else None
