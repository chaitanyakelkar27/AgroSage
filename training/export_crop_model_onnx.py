"""
AgroSage — Export Crop Model to ONNX Format
=============================================
Converts the trained Random Forest (crop_model.pkl) to ONNX format
for portable, dependency-light inference without requiring sklearn.

Usage:
    python training/export_crop_model_onnx.py

Output:
    models/crop_model.onnx
"""

import os
import sys
import time

import joblib
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

PKL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "crop_scaler.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "crop_classes.pkl")
ONNX_PATH = os.path.join(MODEL_DIR, "crop_model.onnx")

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def _separator(title: str) -> None:
    """Print a styled section header."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def main():
    print("\nAgroSage — Crop Model ONNX Export\n")

    # ── 1. Load sklearn model ────────────────────────────────────────────────
    _separator("1 · Loading sklearn artifacts")

    for path, name in [(PKL_PATH, "crop_model.pkl"), (SCALER_PATH, "crop_scaler.pkl")]:
        if not os.path.isfile(path):
            print(f"   ERROR: {name} not found at {path}")
            print("   Run: python training/train_crop_model.py first")
            sys.exit(1)

    model = joblib.load(PKL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = joblib.load(CLASSES_PATH) if os.path.isfile(CLASSES_PATH) else None

    print(f"   Model type    : {type(model).__name__}")
    print(f"   Features      : {len(FEATURE_NAMES)}")
    if classes:
        print(f"   Classes       : {len(classes)}")

    pkl_size = os.path.getsize(PKL_PATH) / 1024
    print(f"   PKL size      : {pkl_size:.1f} KB")

    # ── 2. Convert to ONNX ───────────────────────────────────────────────────
    _separator("2 · Converting to ONNX")

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("   ERROR: skl2onnx not installed. Run: pip install skl2onnx")
        sys.exit(1)

    # Define input type
    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]

    t0 = time.time()
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
        options={type(model): {"zipmap": False}},  # Return array instead of dict
    )
    elapsed = time.time() - t0

    print(f"   Conversion    : {elapsed:.2f}s")

    # Save
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    onnx_size = os.path.getsize(ONNX_PATH) / 1024
    print(f"   ONNX size     : {onnx_size:.1f} KB")
    print(f"   Size ratio    : {onnx_size / pkl_size:.2f}x")
    print(f"   Saved to      : {ONNX_PATH}")

    # ── 3. Validate with ONNX Runtime ────────────────────────────────────────
    _separator("3 · Validating with ONNX Runtime")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(ONNX_PATH)
        input_name = session.get_inputs()[0].name

        # Test input: scaled sample
        test_raw = np.array([[50, 53, 48, 25.0, 70.0, 6.5, 100.0]], dtype=np.float32)
        test_scaled = scaler.transform(test_raw).astype(np.float32)

        # ONNX prediction
        onnx_outputs = session.run(None, {input_name: test_scaled})
        onnx_pred = onnx_outputs[0][0]  # Class label or index

        # sklearn prediction
        sklearn_pred = model.predict(test_scaled)[0]

        print(f"   sklearn pred  : {sklearn_pred}")
        print(f"   ONNX pred     : {onnx_pred}")
        print(f"   Match         : {'✓ Yes' if str(onnx_pred) == str(sklearn_pred) else '✗ No'}")

        # Compare probabilities
        sklearn_proba = model.predict_proba(test_scaled)[0]
        onnx_proba = onnx_outputs[1][0] if len(onnx_outputs) > 1 else None

        if onnx_proba is not None:
            # ONNX may return dict or array
            if isinstance(onnx_proba, dict):
                onnx_proba_arr = np.array([onnx_proba.get(i, 0.0) for i in range(len(sklearn_proba))])
            else:
                onnx_proba_arr = np.array(onnx_proba)

            max_diff = float(np.max(np.abs(sklearn_proba - onnx_proba_arr)))
            print(f"   Max prob diff : {max_diff:.6f}")

    except ImportError:
        print("   WARNING: onnxruntime not installed. Skipping validation.")
        print("   Run: pip install onnxruntime")

    # ── Done ─────────────────────────────────────────────────────────────────
    _separator("Export complete")
    print(f"   Output        : {ONNX_PATH}")
    print(f"   Size          : {onnx_size:.1f} KB")
    print()


if __name__ == "__main__":
    main()
