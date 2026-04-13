"""
AgroSage - Disease Prediction Utility

Reusable helpers for Streamlit disease detection inference.
"""

import os
from typing import Any

import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

_CLASS_MAP = "disease_classes.pkl"
_AUTOENCODER = "autoencoder.h5"
_CNN = "disease_cnn.h5"


def load_disease_models(model_dir: str | None = None) -> dict[str, Any]:
    """Load disease inference artifacts from models directory."""
    directory = model_dir or _MODEL_DIR

    class_map_path = os.path.join(directory, _CLASS_MAP)
    auto_path = os.path.join(directory, _AUTOENCODER)
    cnn_path = os.path.join(directory, _CNN)

    for path in (class_map_path, auto_path, cnn_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing disease artifact: {path}. "
                "Run python training/train_disease_model.py to generate all models."
            )

    class_map = joblib.load(class_map_path)
    if isinstance(class_map, dict) and "idx_to_class" in class_map:
        idx_to_class = {int(k): v for k, v in class_map["idx_to_class"].items()}
    elif isinstance(class_map, dict):
        idx_to_class = {int(v): k for k, v in class_map.items()}
    else:
        raise ValueError("Invalid disease class mapping format in disease_classes.pkl")

    autoencoder = load_model(auto_path, compile=False)
    cnn = load_model(cnn_path, compile=False)

    return {
        "autoencoder": autoencoder,
        "cnn": cnn,
        "idx_to_class": idx_to_class,
    }


def preprocess_image(pil_image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """Convert PIL image to normalized model-ready batch."""
    if not isinstance(pil_image, Image.Image):
        raise TypeError("preprocess_image expects a PIL.Image.Image input")

    image = pil_image.convert("RGB").resize(target_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def clean_image(autoencoder, img_batch: np.ndarray) -> np.ndarray:
    """Run autoencoder cleaner and return clipped normalized image batch."""
    cleaned = autoencoder.predict(img_batch, verbose=0)
    return np.clip(cleaned, 0.0, 1.0)


def classifier_uses_embedded_autoencoder(cnn) -> bool:
    """Return True when classifier graph already contains the autoencoder block."""
    return any(getattr(layer, "name", "") == "plant_autoencoder" for layer in cnn.layers)


def prepare_classifier_input(cnn, autoencoder, img_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (classifier_input, cleaned_preview_batch).

    If the classifier already embeds the autoencoder, raw preprocessed images are used
    for prediction to avoid denoising twice.
    """
    cleaned = clean_image(autoencoder, img_batch)
    if classifier_uses_embedded_autoencoder(cnn):
        return img_batch, cleaned
    return cleaned, cleaned


def parse_disease_label(raw_label: str) -> tuple[str, str]:
    """
    Parse labels like 'Tomato___Late_blight' or 'Tomato___healthy'.
    Returns (plant, disease) with readable text.
    """
    if not raw_label:
        return "Unknown", "Unknown"

    label = raw_label.strip()

    if "___" in label:
        plant_raw, disease_raw = label.split("___", 1)
    else:
        plant_raw, disease_raw = label, "unknown"

    plant = plant_raw.replace("_", " ").strip() or "Unknown"

    disease_token = disease_raw.strip()
    if disease_token.lower() == "healthy":
        disease = "Healthy"
    else:
        disease = disease_token.replace("_", " ").strip() or "Unknown"

    return plant, disease


def predict_disease(cnn, cleaned_batch: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, Any]:
    """Predict disease class and return structured output for UI consumption."""
    probs = cnn.predict(cleaned_batch, verbose=0)[0]
    sorted_indices = np.argsort(probs)[::-1]

    top_idx = int(sorted_indices[0])
    top_label = idx_to_class[top_idx]
    plant, disease = parse_disease_label(top_label)
    probability = float(probs[top_idx])

    if disease.lower() == "healthy":
        alert_level = "none"
    elif probability >= 0.85:
        alert_level = "high"
    elif probability >= 0.60:
        alert_level = "medium"
    else:
        alert_level = "low"

    all_results: list[dict[str, Any]] = []
    for idx in sorted_indices:
        class_label = idx_to_class[int(idx)]
        class_plant, class_disease = parse_disease_label(class_label)
        all_results.append(
            {
                "raw_label": class_label,
                "plant": class_plant,
                "disease": class_disease,
                "probability": float(probs[int(idx)]),
            }
        )

    return {
        "top_disease_raw": top_label,
        "plant": plant,
        "disease": disease,
        "probability": probability,
        "alert_level": alert_level,
        "all_results": all_results,
    }
