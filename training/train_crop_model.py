"""
AgroSage — Crop Recommendation Model Training Pipeline
=======================================================
Trains a Random Forest classifier (with GridSearchCV hyper-parameter tuning)
on the Crop_recommendation.csv dataset.  Also fits a KMeans clustering model
on the scaled feature space so the prediction utility can suggest similar crops.

Artifacts saved to  models/:
    crop_model.pkl      – best Random Forest estimator
    crop_scaler.pkl     – fitted StandardScaler
    crop_classes.pkl    – ordered list of crop label strings
    crop_kmeans.pkl     – fitted KMeans model
    crop_cluster_map.pkl – dict  {cluster_id: [crop1, crop2, …]}
"""

import os
import time
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"


def _separator(title: str) -> None:
    """Print a styled section header."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ── 1. Load & inspect ─────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    _separator("1 · Loading dataset")
    df = pd.read_csv(DATA_PATH)
    # Strip whitespace from column names and label values
    df.columns = df.columns.str.strip()
    df[TARGET_COL] = df[TARGET_COL].str.strip()
    print(f"   Rows          : {len(df):,}")
    print(f"   Features      : {FEATURE_COLS}")
    print(f"   Unique crops  : {df[TARGET_COL].nunique()}")
    print(f"   Crops         : {sorted(df[TARGET_COL].unique())}")
    print(f"   Null values   : {df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()}")
    return df


# ── 2. Split & scale ──────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    _separator("2 · Train / test split & scaling")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"   Train size    : {X_train.shape[0]:,}")
    print(f"   Test  size    : {X_test.shape[0]:,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    print("   Scaler        : StandardScaler fitted")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ── 3. Train with GridSearchCV ─────────────────────────────────────────────────

def train_model(X_train: np.ndarray, y_train: np.ndarray):
    _separator("3 · Training Random Forest + GridSearchCV")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"   Search time   : {elapsed:.1f}s")
    print(f"   Best CV score : {grid.best_score_:.4f}")
    print(f"   Best params   : {grid.best_params_}")

    return grid.best_estimator_


# ── 4. Evaluate ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    _separator("4 · Evaluation on hold-out test set")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Test accuracy : {acc:.4f}  ({acc * 100:.2f}%)\n")
    print(classification_report(y_test, y_pred))
    return acc


# ── 5. Build KMeans cluster map ────────────────────────────────────────────────

def build_cluster_map(df: pd.DataFrame, scaler: StandardScaler, n_clusters: int = 7):
    """
    Cluster the *entire* scaled dataset with KMeans and build a mapping
        {cluster_id: [list of crops ordered by frequency within that cluster]}
    This lets the prediction utility say "crops similar to your recommendation".
    """
    _separator("5 · Building KMeans cluster map")

    X_all = scaler.transform(df[FEATURE_COLS].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_all)

    cluster_map: dict[int, list[str]] = {}
    for cid in range(n_clusters):
        mask = labels == cid
        crops_in_cluster = df.loc[mask, TARGET_COL].values
        freq = Counter(crops_in_cluster)
        # Sort by frequency descending
        cluster_map[cid] = [crop for crop, _ in freq.most_common()]

    print(f"   Clusters      : {n_clusters}")
    for cid, crops in sorted(cluster_map.items()):
        print(f"   Cluster {cid}     : {crops}")

    return kmeans, cluster_map


# ── 6. Save artefacts ─────────────────────────────────────────────────────────

def save_artefacts(model, scaler, crop_classes, kmeans, cluster_map):
    _separator("6 · Saving model artefacts")
    os.makedirs(MODEL_DIR, exist_ok=True)

    paths = {
        "crop_model.pkl": model,
        "crop_scaler.pkl": scaler,
        "crop_classes.pkl": crop_classes,
        "crop_kmeans.pkl": kmeans,
        "crop_cluster_map.pkl": cluster_map,
    }

    for fname, obj in paths.items():
        full = os.path.join(MODEL_DIR, fname)
        joblib.dump(obj, full)
        size_kb = os.path.getsize(full) / 1024
        print(f"   {fname:<25s}  {size_kb:>8.1f} KB")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\nAgroSage — Crop Recommendation Model Training\n")

    # 1. data
    df = load_data()

    # 2. split + scale
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # 3. train
    model = train_model(X_train, y_train)

    # 4. evaluate
    accuracy = evaluate_model(model, X_test, y_test)

    # 5. cluster map
    kmeans, cluster_map = build_cluster_map(df, scaler)

    # 6. save
    crop_classes = sorted(df[TARGET_COL].unique().tolist())
    save_artefacts(model, scaler, crop_classes, kmeans, cluster_map)

    # Done
    _separator("Training complete")
    print(f"   Final test accuracy : {accuracy * 100:.2f}%")
    print(f"   Artefacts saved to  : {MODEL_DIR}")
    print()


if __name__ == "__main__":
    main()
