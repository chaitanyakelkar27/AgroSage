"""
Generate report-ready disease-model figures for AgroSage.

Outputs:
    artifacts/report/loss_graph.png
    artifacts/report/confusion_matrix.png
    artifacts/report/disease_metrics.json

Notes:
- A full historical training-loss file is not stored in this repo, so this script
  runs a short training pass to produce a practical loss curve for reporting.
- Confusion matrix is generated from validation predictions.
"""

import argparse
import json
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "plant_disease_dataset", "train")
VALID_DIR = os.path.join(BASE_DIR, "data", "plant_disease_dataset", "valid")
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts", "report")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLASS_MAP_PATH = os.path.join(MODEL_DIR, "disease_classes.pkl")
CNN_PATH = os.path.join(MODEL_DIR, "disease_cnn.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate loss graph and confusion matrix")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for short loss-curve training pass")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Square image size")
    parser.add_argument("--train-steps", type=int, default=60, help="Steps per epoch for loss-curve pass")
    parser.add_argument("--val-steps", type=int, default=20, help="Validation steps for loss-curve pass")
    parser.add_argument(
        "--conf-steps",
        type=int,
        default=0,
        help="Validation steps for confusion matrix (0 means full validation set)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_generators(img_size: int, batch_size: int):
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valid_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    valid = valid_gen.flow_from_directory(
        VALID_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=42,
    )

    return train, valid


def load_disease_models_local() -> dict:
    if not os.path.isfile(CLASS_MAP_PATH):
        raise FileNotFoundError(f"Missing artifact: {CLASS_MAP_PATH}")
    if not os.path.isfile(CNN_PATH):
        raise FileNotFoundError(f"Missing artifact: {CNN_PATH}")

    class_map = joblib.load(CLASS_MAP_PATH)
    if isinstance(class_map, dict) and "idx_to_class" in class_map:
        idx_to_class = {int(k): v for k, v in class_map["idx_to_class"].items()}
    elif isinstance(class_map, dict):
        idx_to_class = {int(v): k for k, v in class_map.items()}
    else:
        raise ValueError("Invalid class map format in disease_classes.pkl")

    cnn = load_model(CNN_PATH, compile=False)
    return {"cnn": cnn, "idx_to_class": idx_to_class}


def plot_loss(history: tf.keras.callbacks.History, output_path: str) -> None:
    losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    epochs = list(range(1, len(losses) + 1))

    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, losses, marker="o", linewidth=2, label="Training Loss")
    plt.plot(epochs, val_losses, marker="o", linewidth=2, label="Validation Loss")
    plt.title("Disease Model Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: str) -> None:
    # Normalize row-wise for readability in multi-class setting.
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig_w = max(14, len(labels) * 0.35)
    fig_h = max(12, len(labels) * 0.35)

    plt.figure(figsize=(fig_w, fig_h), dpi=180)
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title("Normalized Confusion Matrix (Validation Set)")
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=6)
    plt.yticks(tick_marks, labels, fontsize=6)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VALID_DIR):
        raise FileNotFoundError("Train/validation directories not found in data/plant_disease_dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/4] Loading trained models...")
    model_bundle = load_disease_models_local()
    cnn = model_bundle["cnn"]
    idx_to_class = model_bundle["idx_to_class"]

    print("[2/4] Building data generators...")
    train_gen, valid_gen = build_generators(args.img_size, args.batch_size)

    print("[3/4] Running short training pass for loss graph...")
    cnn.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    steps_per_epoch = min(args.train_steps, len(train_gen))
    val_steps = min(args.val_steps, len(valid_gen))

    history = cnn.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=1,
    )

    loss_graph_path = os.path.join(OUTPUT_DIR, "loss_graph.png")
    plot_loss(history, loss_graph_path)

    print("[4/4] Computing confusion matrix on validation data...")
    # Reload original model to avoid confusion matrix being influenced by short tuning pass.
    model_bundle_eval = load_disease_models_local()
    eval_cnn = model_bundle_eval["cnn"]
    eval_idx_to_class = model_bundle_eval["idx_to_class"]

    _, valid_eval = build_generators(args.img_size, args.batch_size)
    conf_steps = len(valid_eval) if args.conf_steps <= 0 else min(args.conf_steps, len(valid_eval))

    preds = eval_cnn.predict(valid_eval, steps=conf_steps, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = valid_eval.classes[: len(y_pred)]

    labels_idx = sorted(eval_idx_to_class.keys())
    label_names = [eval_idx_to_class[i] for i in labels_idx]

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, label_names, cm_path)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "validation_samples": int(len(y_true)),
        "accuracy": round(accuracy, 6),
        "accuracy_percent": round(accuracy * 100.0, 2),
        "macro_precision": round(float(precision), 6),
        "macro_recall": round(float(recall), 6),
        "macro_f1": round(float(f1), 6),
        "loss_curve_epochs": int(len(history.history.get("loss", []))),
        "loss_curve_note": "Generated from short fine-tuning pass for reporting.",
    }

    metrics_path = os.path.join(OUTPUT_DIR, "disease_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nGenerated artifacts:")
    print(f"- {loss_graph_path}")
    print(f"- {cm_path}")
    print(f"- {metrics_path}")
    print("\nMacro metrics:")
    print(f"- Accuracy:  {metrics['accuracy_percent']}%")
    print(f"- Precision: {metrics['macro_precision']}")
    print(f"- Recall:    {metrics['macro_recall']}")
    print(f"- F1-score:  {metrics['macro_f1']}")


if __name__ == "__main__":
    main()
