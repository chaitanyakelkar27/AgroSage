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

# ── Paths ──────────────────────────────────────────────────────────────────────
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

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


def _current_signature() -> tuple[float, ...]:
    """Return mtimes for all crop artefacts to detect on-disk updates."""
    sig = []
    for fname in _ARTEFACT_NAMES:
        path = os.path.join(_MODEL_DIR, fname)
        if not os.path.isfile(path):
            return ()
        sig.append(os.path.getmtime(path))
    return tuple(sig)


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

        # Top-3 alternatives (excluding the best)
        alternatives = [
            {"name": name, "probability": prob}
            for name, prob in sorted_crops[1:4]
        ]

        # 4. similar crops via KMeans cluster
        cluster_id = int(kmeans.predict(X_scaled)[0])
        similar = cluster_map.get(cluster_id, [])
        # Exclude the best crop itself from the similar list
        similar_crops = [c for c in similar if c != best_crop]

        return {
            "best_crop": best_crop,
            "confidence": confidence,
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
