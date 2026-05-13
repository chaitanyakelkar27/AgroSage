"""
AgroSage — Export Disease Models to TFLite Format
===================================================
Converts the trained Keras models to TensorFlow Lite format
for lightweight on-device inference without full TensorFlow.

Exports:
    disease_cnn.h5      → models/disease_cnn.tflite
    autoencoder.h5      → models/autoencoder.tflite

Usage:
    python training/export_disease_tflite.py
"""

import os
import sys
import time

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CNN_H5_PATH = os.path.join(MODEL_DIR, "disease_cnn.h5")
CNN_TFLITE_PATH = os.path.join(MODEL_DIR, "disease_cnn.tflite")

AUTOENCODER_H5_PATH = os.path.join(MODEL_DIR, "autoencoder.h5")
AUTOENCODER_TFLITE_PATH = os.path.join(MODEL_DIR, "autoencoder.tflite")


def _separator(title: str) -> None:
    """Print a styled section header."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def convert_keras_to_tflite(
    h5_path: str,
    tflite_path: str,
    model_name: str,
) -> bool:
    """
    Convert a Keras .h5 model to TFLite format.

    Parameters
    ----------
    h5_path : str
        Path to the source .h5 Keras model.
    tflite_path : str
        Path to save the output .tflite model.
    model_name : str
        Display name for logging.

    Returns
    -------
    bool
        True if conversion succeeded, False otherwise.
    """
    if not os.path.isfile(h5_path):
        print(f"   SKIP: {model_name} not found at {h5_path}")
        return False

    h5_size = os.path.getsize(h5_path) / 1024
    print(f"   Source        : {h5_path}")
    print(f"   H5 size       : {h5_size:.1f} KB")

    try:
        import tensorflow as tf

        # Load Keras model
        model = tf.keras.models.load_model(h5_path, compile=False)
        print(f"   Input shape   : {model.input_shape}")
        print(f"   Output shape  : {model.output_shape}")
        print(f"   Parameters    : {model.count_params():,}")

        # Convert to TFLite
        t0 = time.time()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply default optimizations for size reduction
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        elapsed = time.time() - t0

        # Save
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        tflite_size = os.path.getsize(tflite_path) / 1024
        print(f"   Conversion    : {elapsed:.2f}s")
        print(f"   TFLite size   : {tflite_size:.1f} KB")
        print(f"   Size ratio    : {tflite_size / h5_size:.2f}x")
        print(f"   Saved to      : {tflite_path}")

        return True

    except Exception as exc:
        print(f"   ERROR: Conversion failed: {type(exc).__name__}: {exc}")
        return False


def validate_tflite(tflite_path: str, model_name: str) -> bool:
    """
    Validate a TFLite model by running a dummy inference.

    Parameters
    ----------
    tflite_path : str
        Path to the .tflite model.
    model_name : str
        Display name for logging.

    Returns
    -------
    bool
        True if validation passed, False otherwise.
    """
    if not os.path.isfile(tflite_path):
        print(f"   SKIP: {model_name} TFLite not found")
        return False

    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]

        print(f"   Input shape   : {input_shape}")
        print(f"   Input dtype   : {input_dtype}")
        print(f"   Output shape  : {output_details[0]['shape']}")

        # Create dummy input
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], dummy_input)

        t0 = time.time()
        interpreter.invoke()
        elapsed = (time.time() - t0) * 1000  # ms

        output = interpreter.get_tensor(output_details[0]["index"])
        print(f"   Inference     : {elapsed:.1f}ms")
        print(f"   Output sample : {output[0][:5]}...")
        print(f"   Validation    : ✓ Passed")

        return True

    except Exception as exc:
        print(f"   Validation failed: {type(exc).__name__}: {exc}")
        return False


def main():
    print("\nAgroSage — Disease Model TFLite Export\n")

    # ── 1. Convert CNN ───────────────────────────────────────────────────────
    _separator("1 · Converting Disease CNN")
    cnn_ok = convert_keras_to_tflite(CNN_H5_PATH, CNN_TFLITE_PATH, "Disease CNN")

    # ── 2. Convert Autoencoder ───────────────────────────────────────────────
    _separator("2 · Converting Autoencoder")
    ae_ok = convert_keras_to_tflite(
        AUTOENCODER_H5_PATH, AUTOENCODER_TFLITE_PATH, "Autoencoder"
    )

    # ── 3. Validate ──────────────────────────────────────────────────────────
    if cnn_ok:
        _separator("3 · Validating Disease CNN TFLite")
        validate_tflite(CNN_TFLITE_PATH, "Disease CNN")

    if ae_ok:
        _separator("4 · Validating Autoencoder TFLite")
        validate_tflite(AUTOENCODER_TFLITE_PATH, "Autoencoder")

    # ── Summary ──────────────────────────────────────────────────────────────
    _separator("Export complete")
    if cnn_ok:
        print(f"   Disease CNN   : {CNN_TFLITE_PATH}")
    if ae_ok:
        print(f"   Autoencoder   : {AUTOENCODER_TFLITE_PATH}")
    if not cnn_ok and not ae_ok:
        print("   No models were exported. Train the models first:")
        print("   python training/train_disease_model.py")
    print()


if __name__ == "__main__":
    main()
