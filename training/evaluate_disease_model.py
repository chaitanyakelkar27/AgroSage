"""
AgroSage - Disease Model Evaluation

Runs a reproducible top-1 accuracy check on validation images using the
same inference path used by the Streamlit app.
"""

import argparse
import glob
import os
import random
import sys
from collections import defaultdict

from PIL import Image


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALID_DIR = os.path.join(BASE_DIR, "data", "plant_disease_dataset", "valid")
IMG_SIZE = (128, 128)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.disease_predictor import (  # noqa: E402
    load_disease_models,
    prepare_classifier_input,
    predict_disease,
    preprocess_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AgroSage disease model")
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Maximum number of validation images to evaluate (default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


def collect_image_paths(valid_dir: str) -> list[tuple[str, str]]:
    class_dirs = [
        d
        for d in sorted(os.listdir(valid_dir))
        if os.path.isdir(os.path.join(valid_dir, d))
    ]

    paths: list[tuple[str, str]] = []
    for class_name in class_dirs:
        folder = os.path.join(valid_dir, class_name)
        images = (
            glob.glob(os.path.join(folder, "*.jpg"))
            + glob.glob(os.path.join(folder, "*.JPG"))
            + glob.glob(os.path.join(folder, "*.jpeg"))
            + glob.glob(os.path.join(folder, "*.JPEG"))
            + glob.glob(os.path.join(folder, "*.png"))
            + glob.glob(os.path.join(folder, "*.PNG"))
            + glob.glob(os.path.join(folder, "*.webp"))
            + glob.glob(os.path.join(folder, "*.WEBP"))
        )
        paths.extend((img_path, class_name) for img_path in images)

    return paths


def main() -> None:
    args = parse_args()

    if not os.path.isdir(VALID_DIR):
        raise FileNotFoundError(f"Validation directory not found: {VALID_DIR}")

    all_items = collect_image_paths(VALID_DIR)
    if not all_items:
        raise RuntimeError("No validation images found.")

    random.seed(args.seed)
    random.shuffle(all_items)
    subset = all_items[: max(1, min(args.samples, len(all_items)))]

    models = load_disease_models()

    total = 0
    correct = 0
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)

    for img_path, true_label in subset:
        with Image.open(img_path) as img:
            batch = preprocess_image(img, IMG_SIZE)

        classifier_input, _ = prepare_classifier_input(
            models["cnn"], models["autoencoder"], batch
        )
        result = predict_disease(models["cnn"], classifier_input, models["idx_to_class"])
        pred_label = result["top_disease_raw"]

        total += 1
        per_class_total[true_label] += 1
        if pred_label == true_label:
            correct += 1
            per_class_correct[true_label] += 1

    accuracy = (correct / total) * 100.0

    print("\nAgroSage Disease Evaluation")
    print("=" * 36)
    print(f"Validation samples checked: {total}")
    print(f"Top-1 exact-label accuracy: {accuracy:.2f}% ({correct}/{total})")

    hardest = []
    for class_name, class_total in per_class_total.items():
        class_correct = per_class_correct.get(class_name, 0)
        class_acc = class_correct / class_total
        hardest.append((class_acc, class_name, class_correct, class_total))

    hardest.sort(key=lambda x: x[0])
    print("\nLowest class accuracies in this sample:")
    for class_acc, class_name, class_correct, class_total in hardest[:10]:
        print(
            f"- {class_name}: {class_acc * 100:.2f}% "
            f"({class_correct}/{class_total})"
        )


if __name__ == "__main__":
    main()
