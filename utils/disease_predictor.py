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

try:
    import torch
    from transformers import AutoImageProcessor, ViTForImageClassification
except Exception:  # noqa: BLE001
    torch = None
    AutoImageProcessor = None
    ViTForImageClassification = None


_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

_CLASS_MAP = "disease_classes.pkl"
_AUTOENCODER = "autoencoder.h5"
_CNN = "disease_cnn.h5"
_VIT_DIR = "vit"

_RECON_ERROR_REVIEW_THRESHOLD = 0.00150
_EDGE_DENSITY_REVIEW_THRESHOLD = 0.0650
_GREEN_RATIO_MIN_THRESHOLD = 0.06
_GREEN_RATIO_STRICT_THRESHOLD = 0.04
_LESION_EVIDENCE_LOW_THRESHOLD = 0.035
_INCONCLUSIVE_HEALTHY_PROB_THRESHOLD = 0.20
_INCONCLUSIVE_GAP_THRESHOLD = 0.30
_TTA_TOP_CONSISTENCY_THRESHOLD = 0.60
_TTA_TOP_STD_THRESHOLD = 0.10


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


def _require_vit_dependencies() -> None:
    if torch is None or AutoImageProcessor is None or ViTForImageClassification is None:
        raise ImportError(
            "ViT dependencies not available. Install torch, torchvision, and transformers."
        )


def load_vit_model(model_dir: str | None = None) -> dict[str, Any]:
    """Load a fine-tuned ViT model from models/vit (transformers format)."""
    _require_vit_dependencies()

    directory = model_dir or _MODEL_DIR
    vit_dir = os.path.join(directory, _VIT_DIR)

    if not os.path.isdir(vit_dir):
        raise FileNotFoundError(
            f"Missing ViT model directory: {vit_dir}. "
            "Place a fine-tuned transformers ViT model in models/vit."
        )

    processor = AutoImageProcessor.from_pretrained(vit_dir)
    model = ViTForImageClassification.from_pretrained(vit_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id_to_label = model.config.id2label or {}
    return {
        "vit": model,
        "vit_processor": processor,
        "vit_id_to_label": id_to_label,
        "vit_device": device,
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
    elif " - " in label:
        plant_raw, disease_raw = label.split(" - ", 1)
    else:
        plant_raw, disease_raw = label, "unknown"

    plant = plant_raw.replace("_", " ").strip() or "Unknown"

    disease_token = disease_raw.strip()
    if disease_token.lower() == "healthy":
        disease = "Healthy"
    else:
        disease = disease_token.replace("_", " ").strip() or "Unknown"

    return plant, disease


def compute_image_quality_metrics(image_batch: np.ndarray) -> dict[str, float]:
    """Compute lightweight quality/domain metrics from a normalized image batch."""
    if image_batch.ndim != 4 or image_batch.shape[0] != 1 or image_batch.shape[-1] != 3:
        raise ValueError("Expected image batch shape (1, H, W, 3)")

    arr = np.asarray(image_batch[0], dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_density = float((dx.mean() + dy.mean()) / 2.0)

    green_mask = (g > 0.20) & (g >= r * 0.9) & (g >= b * 0.9)
    green_ratio = float(green_mask.mean())

    # Lightweight color-based proxy for visible lesion evidence.
    # This is not a medical diagnosis, only a sanity check for false positives.
    brown_mask = (r > 0.28) & (r > g * 1.15) & (g > 0.10) & (g < 0.45) & (b < 0.35)
    yellow_mask = (r > 0.45) & (g > 0.45) & (b < 0.40) & (np.abs(r - g) < 0.22)
    dark_lesion_mask = (gray < 0.22) & (r > b)

    brown_ratio = float(brown_mask.mean())
    yellow_ratio = float(yellow_mask.mean())
    dark_lesion_ratio = float(dark_lesion_mask.mean())
    lesion_evidence_score = float((brown_ratio * 1.0) + (yellow_ratio * 0.8) + (dark_lesion_ratio * 0.4))

    return {
        "edge_density": edge_density,
        "green_ratio": green_ratio,
        "brightness": float(gray.mean()),
        "brown_ratio": brown_ratio,
        "yellow_ratio": yellow_ratio,
        "dark_lesion_ratio": dark_lesion_ratio,
        "lesion_evidence_score": lesion_evidence_score,
    }


def _is_lesion_type_disease(disease_name: str) -> bool:
    """Return True when disease typically shows visible lesions/spots on leaves."""
    name = disease_name.strip().lower()
    keywords = ("spot", "blight", "rot", "mold", "rust", "scab", "scorch")
    return any(token in name for token in keywords)


def _build_tta_batch(image_batch: np.ndarray) -> np.ndarray:
    """Build a small test-time augmentation batch for robust field inference."""
    if image_batch.ndim != 4 or image_batch.shape[0] != 1 or image_batch.shape[-1] != 3:
        raise ValueError("Expected image batch shape (1, H, W, 3)")

    img = np.asarray(image_batch[0], dtype=np.float32)
    variants = [
        img,
        np.flip(img, axis=1),
        np.clip(img * 0.92, 0.0, 1.0),
        np.clip(img * 1.08, 0.0, 1.0),
        np.clip(np.flip(img, axis=1) * 1.04, 0.0, 1.0),
    ]
    return np.stack(variants, axis=0).astype(np.float32)


def _healthy_probability_for_plant(
    probs: np.ndarray,
    idx_to_class: dict[int, str],
    plant_name: str,
) -> float:
    """Return probability of '<plant>___healthy' when available."""
    plant_key = plant_name.strip().lower()
    for idx, label in idx_to_class.items():
        class_plant, class_disease = parse_disease_label(label)
        if class_plant.strip().lower() == plant_key and class_disease.strip().lower() == "healthy":
            return float(probs[int(idx)])
    return 0.0


def _align_probs_with_class_map(
    probs: np.ndarray,
    vit_id_to_label: dict[int, str],
    idx_to_class: dict[int, str],
) -> np.ndarray:
    if not vit_id_to_label:
        return probs

    label_to_prob: dict[str, float] = {}
    for idx, label in vit_id_to_label.items():
        label_to_prob[str(label).strip().lower()] = float(probs[int(idx)])

    aligned = []
    for idx in sorted(idx_to_class.keys()):
        label_key = idx_to_class[idx].strip().lower()
        aligned.append(label_to_prob.get(label_key, 0.0))
    return np.asarray(aligned, dtype=np.float32)


def _build_prediction_output(
    probs: np.ndarray,
    idx_to_class: dict[int, str],
    reconstruction_error: float | None,
    image_quality_metrics: dict[str, float] | None,
    inference_mode: str,
    tta_probs: np.ndarray | None = None,
) -> dict[str, Any]:
    mode = (inference_mode or "field").strip().lower()
    if mode not in {"field", "standard"}:
        mode = "field"

    if tta_probs is None:
        tta_probs = np.expand_dims(probs, axis=0)

    sorted_indices = np.argsort(probs)[::-1]

    top_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_idx
    top_label = idx_to_class[top_idx]
    plant, disease = parse_disease_label(top_label)
    probability = float(probs[top_idx])
    second_probability = float(probs[second_idx])
    confidence_gap = max(0.0, probability - second_probability)
    healthy_probability = _healthy_probability_for_plant(probs, idx_to_class, plant)
    top_consistency = float(np.mean(np.argmax(tta_probs, axis=1) == top_idx))
    top_probability_std = float(np.std(tta_probs[:, top_idx]))

    quality_metrics = image_quality_metrics or {}
    edge_density = float(quality_metrics.get("edge_density", 0.0))
    green_ratio = float(quality_metrics.get("green_ratio", 1.0))
    lesion_evidence_score = float(quality_metrics.get("lesion_evidence_score", 1.0))

    diagnostic_flags: list[str] = []
    guardrail_score = 0
    if reconstruction_error is not None and reconstruction_error > _RECON_ERROR_REVIEW_THRESHOLD:
        diagnostic_flags.append("high_reconstruction_error")
        guardrail_score += 1
    if edge_density > _EDGE_DENSITY_REVIEW_THRESHOLD:
        diagnostic_flags.append("high_background_complexity")
        guardrail_score += 1
    if green_ratio < _GREEN_RATIO_STRICT_THRESHOLD:
        diagnostic_flags.append("very_low_leaf_coverage")
        guardrail_score += 2
    elif green_ratio < _GREEN_RATIO_MIN_THRESHOLD:
        diagnostic_flags.append("low_leaf_coverage")
        guardrail_score += 1

    likely_false_positive = False
    if disease.lower() != "healthy" and _is_lesion_type_disease(disease):
        if lesion_evidence_score < _LESION_EVIDENCE_LOW_THRESHOLD:
            diagnostic_flags.append("weak_lesion_evidence")
            guardrail_score += 2
            likely_false_positive = True

    inconclusive_reasons: list[str] = []
    if disease.lower() != "healthy":
        if likely_false_positive:
            inconclusive_reasons.append("weak_lesion_evidence")
        if healthy_probability >= _INCONCLUSIVE_HEALTHY_PROB_THRESHOLD and confidence_gap < _INCONCLUSIVE_GAP_THRESHOLD:
            inconclusive_reasons.append("healthy_competition")
        if mode == "field" and top_consistency < _TTA_TOP_CONSISTENCY_THRESHOLD:
            inconclusive_reasons.append("unstable_tta_prediction")
        if mode == "field" and top_probability_std > _TTA_TOP_STD_THRESHOLD:
            inconclusive_reasons.append("high_tta_variance")

    is_inconclusive = len(inconclusive_reasons) > 0
    for reason in inconclusive_reasons:
        if reason not in diagnostic_flags:
            diagnostic_flags.append(reason)

    # Guardrail is enabled only when multiple weak signals combine.
    # This keeps normal field photos (hands/background present) from over-triggering.
    requires_manual_review = (guardrail_score >= 2) or is_inconclusive

    if disease.lower() == "healthy":
        alert_level = "none"
    elif is_inconclusive:
        alert_level = "low"
    elif requires_manual_review:
        alert_level = "medium" if probability >= 0.90 else "low"
    elif probability >= 0.96 and confidence_gap >= 0.65 and healthy_probability < 0.03:
        alert_level = "high"
    elif probability >= 0.75 and confidence_gap >= 0.20:
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
        "second_probability": second_probability,
        "confidence_gap": confidence_gap,
        "healthy_probability_for_plant": healthy_probability,
        "top_consistency": top_consistency,
        "top_probability_std": top_probability_std,
        "is_inconclusive": is_inconclusive,
        "inference_mode": mode,
        "alert_level": alert_level,
        "requires_manual_review": requires_manual_review,
        "likely_false_positive": likely_false_positive,
        "diagnostic_flags": diagnostic_flags,
        "reconstruction_error": reconstruction_error,
        "image_quality_metrics": quality_metrics,
        "all_results": all_results,
    }


def predict_disease(
    cnn,
    cleaned_batch: np.ndarray,
    idx_to_class: dict[int, str],
    reconstruction_error: float | None = None,
    image_quality_metrics: dict[str, float] | None = None,
    inference_mode: str = "field",
) -> dict[str, Any]:
    """Predict disease class and return structured output for UI consumption."""
    mode = (inference_mode or "field").strip().lower()
    if mode not in {"field", "standard"}:
        mode = "field"

    if mode == "field":
        tta_batch = _build_tta_batch(cleaned_batch)
        tta_probs = cnn.predict(tta_batch, verbose=0)
        probs = np.mean(tta_probs, axis=0)
    else:
        probs = cnn.predict(cleaned_batch, verbose=0)[0]
        tta_probs = np.expand_dims(probs, axis=0)

    return _build_prediction_output(
        probs=probs,
        idx_to_class=idx_to_class,
        reconstruction_error=reconstruction_error,
        image_quality_metrics=image_quality_metrics,
        inference_mode=mode,
        tta_probs=tta_probs,
    )


def predict_disease_vit(
    vit,
    vit_processor,
    vit_id_to_label: dict[int, str],
    pil_image: Image.Image,
    idx_to_class: dict[int, str],
    reconstruction_error: float | None = None,
    image_quality_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run ViT inference and return structured output aligned to class map."""
    _require_vit_dependencies()

    inputs = vit_processor(images=pil_image, return_tensors="pt")
    device = next(vit.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = vit(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    aligned_probs = _align_probs_with_class_map(probs, vit_id_to_label, idx_to_class)

    return _build_prediction_output(
        probs=aligned_probs,
        idx_to_class=idx_to_class,
        reconstruction_error=reconstruction_error,
        image_quality_metrics=image_quality_metrics,
        inference_mode="standard",
        tta_probs=np.expand_dims(aligned_probs, axis=0),
    )
