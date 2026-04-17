"""
AgroSage — Crop Prediction Utility
====================================
Loads the trained model artefacts and exposes:

    load_crop_models()    → dict of model objects (cached)
    predict_crop(...)     → dict ready for the Streamlit UI

The return dict contains:
    best_crop        – str
    confidence       – float   (0-100 %)
    alternatives     – list[dict]  top-3 alternatives with name & probability
    similar_crops    – list[str]   crops in the same KMeans cluster
    all_probabilities – dict[str, float]  {crop_name: probability %}
"""

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "Crop_recommendation.csv")

_PROFILE_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
_FEATURE_LABELS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}

_ARTEFACT_NAMES = [
    "crop_model.pkl",
    "crop_scaler.pkl",
    "crop_classes.pkl",
    "crop_kmeans.pkl",
    "crop_cluster_map.pkl",
]

# Module-level cache so we only load once per process
_cache: dict[str, Any] | None = None
_cache_signature: tuple[float, ...] | None = None
_profile_cache: dict[str, dict[str, dict[str, float]]] | None = None
_profile_cache_mtime: float | None = None


def _current_signature() -> tuple[float, ...]:
    """Return mtimes for all crop artefacts to detect on-disk updates."""
    sig = []
    for fname in _ARTEFACT_NAMES:
        path = os.path.join(_MODEL_DIR, fname)
        if not os.path.isfile(path):
            return ()
        sig.append(os.path.getmtime(path))
    return tuple(sig)


def _season_hint(temperature: float, humidity: float, rainfall: float, crop: str) -> str:
    """Return a short human-readable seasonal context hint."""
    if temperature >= 30 and humidity >= 70 and rainfall >= 160:
        return (
            f"The warm, humid, high-rainfall profile resembles a monsoon window where {crop} can perform well."
        )
    if temperature >= 30 and rainfall < 120:
        return (
            f"The hot and relatively dry profile resembles a summer window that aligns with {crop}."
        )
    if temperature < 20:
        return (
            f"The cooler temperature profile resembles a cool-season window that can suit {crop}."
        )
    return (
        f"The moderate climate profile is compatible with common growing conditions for {crop}."
    )


def _load_crop_profiles() -> dict[str, dict[str, dict[str, float]]]:
    """Build robust per-crop feature ranges from the crop dataset for explainability."""
    global _profile_cache, _profile_cache_mtime

    if not os.path.isfile(_DATA_PATH):
        return {}

    current_mtime = os.path.getmtime(_DATA_PATH)
    if _profile_cache is not None and _profile_cache_mtime == current_mtime:
        return _profile_cache

    try:
        df = pd.read_csv(_DATA_PATH)
        df.columns = df.columns.str.strip()
        if "label" not in df.columns:
            return {}
        df["label"] = df["label"].astype(str).str.strip()

        if any(col not in df.columns for col in _PROFILE_FEATURES):
            return {}

        profiles: dict[str, dict[str, dict[str, float]]] = {}
        grouped = df.groupby("label", sort=True)

        for crop, group in grouped:
            crop_profile: dict[str, dict[str, float]] = {}
            for feature in _PROFILE_FEATURES:
                series = pd.to_numeric(group[feature], errors="coerce").dropna()
                if series.empty:
                    continue
                q1 = float(series.quantile(0.25))
                median = float(series.quantile(0.50))
                q3 = float(series.quantile(0.75))
                crop_profile[feature] = {
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                }
            if crop_profile:
                profiles[crop] = crop_profile

        _profile_cache = profiles
        _profile_cache_mtime = current_mtime
        return profiles
    except Exception:  # noqa: BLE001
        return {}


def _feature_match_reason(
    crop: str,
    feature: str,
    value: float,
    stats: dict[str, float],
) -> tuple[float, str]:
    """Return (match_score, reason_text) for a single feature."""
    q1 = stats["q1"]
    q3 = stats["q3"]
    median = stats["median"]

    iqr = max(q3 - q1, 1e-6)
    deviation = abs(value - median) / iqr
    # 1.0 is excellent match; 0.0 means far from historical center.
    match_score = max(0.0, 1.0 - min(deviation, 2.0) / 2.0)

    label = _FEATURE_LABELS.get(feature, feature)
    if q1 <= value <= q3:
        reason = (
            f"{label} ({value:.1f}) is within the typical {crop} range ({q1:.1f}-{q3:.1f})."
        )
    elif deviation <= 1.5:
        reason = (
            f"{label} ({value:.1f}) is close to the usual {crop} range ({q1:.1f}-{q3:.1f})."
        )
    else:
        reason = (
            f"{label} ({value:.1f}) is outside the common {crop} range ({q1:.1f}-{q3:.1f}),"
            " so this is a weaker match."
        )
        reason = "".join(reason)

    return match_score, reason


def _build_crop_reasons(
    crop: str,
    probability: float,
    rank: int,
    feature_values: dict[str, float],
    profiles: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    """Generate concise natural-language reasons for one crop recommendation."""
    crop_profile = profiles.get(crop)
    if not crop_profile:
        return [
            f"Ranked #{rank} by the model with {probability:.1f}% confidence.",
            "Detailed agronomic matching is unavailable because crop profile statistics were not found.",
        ]

    feature_reasons: list[tuple[float, str]] = []
    for feature, value in feature_values.items():
        stats = crop_profile.get(feature)
        if not stats:
            continue
        feature_reasons.append(_feature_match_reason(crop, feature, value, stats))

    feature_reasons.sort(key=lambda item: item[0], reverse=True)
    top_reasons = [reason for _, reason in feature_reasons[:3]]

    climate_line = _season_hint(
        temperature=feature_values["temperature"],
        humidity=feature_values["humidity"],
        rainfall=feature_values["rainfall"],
        crop=crop,
    )

    return [
        f"Ranked #{rank} with model confidence {probability:.1f}% for the current input profile.",
        *top_reasons,
        climate_line,
    ]


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_crop_models() -> dict[str, Any]:
    """
    Load and cache all crop-model artefacts from ``models/``.

    Returns
    -------
    dict with keys:
        model, scaler, classes, kmeans, cluster_map
    """
    global _cache, _cache_signature
    current_sig = _current_signature()
    if _cache is not None and _cache_signature == current_sig:
        return _cache

    artefacts: dict[str, Any] = {}
    keys = ["model", "scaler", "classes", "kmeans", "cluster_map"]

    for key, fname in zip(keys, _ARTEFACT_NAMES):
        path = os.path.join(_MODEL_DIR, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing artefact: {path}\n"
                "Run  python training/train_crop_model.py  to generate model files."
            )
        artefacts[key] = joblib.load(path)

    _cache = artefacts
    _cache_signature = _current_signature()
    return _cache


# ── Predictor ──────────────────────────────────────────────────────────────────

def predict_crop(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
) -> dict[str, Any]:
    """
    Predict the best crop for the given soil / weather conditions.

    Parameters
    ----------
    nitrogen, phosphorus, potassium : float
        N-P-K values (mg/kg).
    temperature : float
        Average temperature (°C).
    humidity : float
        Relative humidity (%).
    ph : float
        Soil pH.
    rainfall : float
        Annual rainfall (mm).

    Returns
    -------
    dict
        best_crop        : str          – top predicted crop
        confidence       : float        – probability % of best crop
        best_reasons     : list[str]    – natural-language reasons for best crop
        alternatives     : list[dict]   – next 3 alternatives [{name, probability}]
        similar_crops    : list[str]    – other crops in the same cluster
        all_probabilities: dict[str,float] – {crop: probability %}

    On any processing error the function returns a dict with key ``error``.
    """
    try:
        # 1. load artefacts
        arts = load_crop_models()
        model = arts["model"]
        scaler = arts["scaler"]
        classes: list[str] = arts["classes"]
        kmeans = arts["kmeans"]
        cluster_map: dict[int, list[str]] = arts["cluster_map"]

        # 2. build input array  (same feature order as training)
        raw = np.array(
            [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
            dtype=np.float64,
        )
        X_scaled = scaler.transform(raw)

        # 3. class probabilities
        proba = model.predict_proba(X_scaled)[0]  # shape (n_classes,)

        # Map to class names
        all_probs: dict[str, float] = {
            crop: round(float(p) * 100, 2) for crop, p in zip(classes, proba)
        }

        # Sort descending
        sorted_crops = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        best_crop = sorted_crops[0][0]
        confidence = sorted_crops[0][1]

        feature_values = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }
        profiles = _load_crop_profiles()
        best_reasons = _build_crop_reasons(
            crop=best_crop,
            probability=confidence,
            rank=1,
            feature_values=feature_values,
            profiles=profiles,
        )

        # Top-3 alternatives (excluding the best)
        alternatives = []
        for rank, (name, prob) in enumerate(sorted_crops[1:4], start=2):
            alternatives.append(
                {
                    "name": name,
                    "probability": prob,
                    "reasons": _build_crop_reasons(
                        crop=name,
                        probability=prob,
                        rank=rank,
                        feature_values=feature_values,
                        profiles=profiles,
                    ),
                }
            )

        # 4. similar crops via KMeans cluster
        cluster_id = int(kmeans.predict(X_scaled)[0])
        similar = cluster_map.get(cluster_id, [])
        # Exclude the best crop itself from the similar list
        similar_crops = [c for c in similar if c != best_crop]

        return {
            "best_crop": best_crop,
            "confidence": confidence,
            "best_reasons": best_reasons,
            "alternatives": alternatives,
            "similar_crops": similar_crops,
            "all_probabilities": all_probs,
        }

    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {
            "error": (
                f"Prediction failed: {type(exc).__name__}: {exc}\n"
                "Please ensure the model artefacts are up-to-date by running:\n"
                "    python training/train_crop_model.py"
            )
        }
