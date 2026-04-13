"""
AgroSage - Plant Disease Detection Training Pipeline

This script trains:
1) A convolutional autoencoder used as an image cleaner.
2) A CNN classifier for plant disease classification.

Artifacts saved to models/:
    disease_classes.pkl
    autoencoder.h5
    disease_cnn.h5
"""

import os
import random

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "plant_disease_dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLASS_MAP_PATH = os.path.join(MODEL_DIR, "disease_classes.pkl")
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder.h5")
CNN_PATH = os.path.join(MODEL_DIR, "disease_cnn.h5")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOENCODER_EPOCHS = 12
CLASSIFIER_EPOCHS = 20
SEED = 42
RESUME_TRAINING = True
BACKUP_DIR = os.path.join(MODEL_DIR, "training_backup")
TRAIN_WORKERS = max(1, min(4, (os.cpu_count() or 1) // 2))
FAST_MODE = True
FAST_CLASSIFIER_EPOCHS = 8
SKIP_AUTOENCODER_IF_EXISTS = True


def _separator(title: str) -> None:
    width = 62
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_paths() -> None:
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    if not os.path.isdir(VALID_DIR):
        raise FileNotFoundError(f"Validation directory not found: {VALID_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)


def create_generators(with_autoencoder_generators: bool = True):
    _separator("1) Building image generators")

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

    train_auto = None
    valid_auto = None
    if with_autoencoder_generators:
        train_auto = train_aug.flow_from_directory(
            TRAIN_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="input",
            shuffle=True,
            seed=SEED,
        )

        valid_auto = valid_gen.flow_from_directory(
            VALID_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="input",
            shuffle=False,
            seed=SEED,
        )

    train_cls = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    valid_cls = valid_gen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
    )

    class_to_idx = train_cls.class_indices
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    print(f"Class count: {len(class_to_idx)}")
    print(f"Train images: {train_cls.samples}")
    print(f"Valid images: {valid_cls.samples}")

    return train_auto, valid_auto, train_cls, valid_cls, class_to_idx, idx_to_class


def save_class_mapping(class_to_idx: dict, idx_to_class: dict) -> None:
    _separator("2) Saving class mapping")
    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    joblib.dump(payload, CLASS_MAP_PATH)
    print(f"Saved: {CLASS_MAP_PATH}")


def build_autoencoder(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="img_input")
    x = layers.GaussianNoise(0.05, name="noise")(inputs)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same", name="encoded")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(3, (3, 3), padding="same", activation="sigmoid", name="cleaned")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="plant_autoencoder")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
        steps_per_execution=32,
    )
    return model


def train_autoencoder(
    autoencoder: tf.keras.Model,
    train_auto,
    valid_auto,
) -> None:
    if train_auto is None or valid_auto is None:
        _separator("3) Skipping autoencoder training")
        print("Using saved autoencoder weights for faster training run.")
        return

    _separator("3) Training autoencoder")
    auto_callbacks = [
        callbacks.BackupAndRestore(backup_dir=BACKUP_DIR),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=AUTOENCODER_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    autoencoder.fit(
        train_auto,
        validation_data=valid_auto,
        epochs=AUTOENCODER_EPOCHS,
        callbacks=auto_callbacks,
        verbose=1,
    )

    autoencoder.save(AUTOENCODER_PATH)
    print(f"Saved: {AUTOENCODER_PATH}")


def get_autoencoder(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    if RESUME_TRAINING and os.path.exists(AUTOENCODER_PATH):
        _separator("3) Loading existing autoencoder")
        print(f"Loaded: {AUTOENCODER_PATH}")
        model = tf.keras.models.load_model(AUTOENCODER_PATH, compile=False)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
        return model
    return build_autoencoder(input_shape=input_shape)


def build_classifier(autoencoder: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    autoencoder.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="classifier_input")
    cleaned = autoencoder(inputs, training=False)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(cleaned)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_logits")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="disease_classifier")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=32,
    )
    return model


def train_classifier(classifier: tf.keras.Model, train_cls, valid_cls) -> float:
    _separator("4) Training classifier")

    cls_callbacks = [
        callbacks.BackupAndRestore(backup_dir=BACKUP_DIR),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=CNN_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    classifier_epochs = FAST_CLASSIFIER_EPOCHS if FAST_MODE else CLASSIFIER_EPOCHS

    history = classifier.fit(
        train_cls,
        validation_data=valid_cls,
        epochs=classifier_epochs,
        callbacks=cls_callbacks,
        verbose=1,
    )

    classifier.save(CNN_PATH)
    print(f"Saved: {CNN_PATH}")

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    return float(best_val_acc)


def get_classifier(autoencoder: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    if RESUME_TRAINING and os.path.exists(CNN_PATH):
        _separator("4) Loading existing classifier")
        print(f"Loaded: {CNN_PATH}")
        model = tf.keras.models.load_model(CNN_PATH, compile=False)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
    return build_classifier(autoencoder=autoencoder, num_classes=num_classes)


def main() -> None:
    print("\nAgroSage - Plant Disease Model Training\n")

    set_reproducibility(SEED)
    ensure_paths()

    should_train_auto = not (
        FAST_MODE and SKIP_AUTOENCODER_IF_EXISTS and os.path.exists(AUTOENCODER_PATH)
    )

    train_auto, valid_auto, train_cls, valid_cls, class_to_idx, idx_to_class = create_generators(
        with_autoencoder_generators=should_train_auto
    )
    save_class_mapping(class_to_idx, idx_to_class)

    autoencoder = get_autoencoder(input_shape=(*IMG_SIZE, 3))
    train_autoencoder(autoencoder, train_auto, valid_auto)

    classifier = get_classifier(autoencoder, num_classes=len(class_to_idx))
    best_val_acc = train_classifier(classifier, train_cls, valid_cls)

    _separator("Training complete")
    print(f"Class count: {len(class_to_idx)}")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Artifacts directory: {MODEL_DIR}\n")


if __name__ == "__main__":
    main()
