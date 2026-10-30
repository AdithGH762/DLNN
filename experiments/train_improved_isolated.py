"""Train the improved CNN-LSTM without accessing held-out test recordings."""
from __future__ import annotations

import json
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import DATA_DIR, MAX_FRAMES, MODEL_DIR, N_MFCC, RANDOM_SEED, SAMPLE_RATE
from src.dataset import discover_ravdess
from src.model import build_cnn_lstm
from src.noise import add_white_noise_at_snr
from src.preprocessing import N_MFCC_DELTA, extract_mfcc_with_deltas, load_audio

EXPERIMENT_DIR = SCRIPT_PATH.parent / "improved_cnn_lstm_run_01"
MODEL_PATH = EXPERIMENT_DIR / "improved_cnn_lstm.keras"
MAPPING_PATH = EXPERIMENT_DIR / "label_mapping.json"
HISTORY_PATH = EXPERIMENT_DIR / "training_history.csv"
CURVE_PATH = EXPERIMENT_DIR / "learning_curve.png"
SUMMARY_PATH = EXPERIMENT_DIR / "experiment_summary.json"

EXPECTED_ACTOR_COUNTS = {"train": 14, "validation": 5, "test": 5}
NOISE_AUGMENTATION = {
    "training_only": True,
    "probability": 0.5,
    "snr_min_db": 5.0,
    "snr_max_db": 30.0,
}


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_validate_split() -> dict[str, list[int]]:
    split_path = MODEL_DIR / "split_actors.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Actor split manifest is missing: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split, dict) or set(split) != {"train", "validation", "test"}:
        raise ValueError("Actor split must contain exactly train, validation, and test.")

    normalized: dict[str, list[int]] = {}
    for name, expected_count in EXPECTED_ACTOR_COUNTS.items():
        actors = split[name]
        if (
            not isinstance(actors, list)
            or any(
                isinstance(actor, bool) or not isinstance(actor, int)
                for actor in actors
            )
            or len(actors) != expected_count
            or len(set(actors)) != len(actors)
            or any(actor < 1 or actor > 24 for actor in actors)
        ):
            raise ValueError(
                f"Actor split '{name}' must contain {expected_count} unique actor IDs."
            )
        normalized[name] = actors

    train = set(normalized["train"])
    validation = set(normalized["validation"])
    test = set(normalized["test"])
    if train & validation or train & test or validation & test:
        raise ValueError("Train, validation, and test actor lists must be disjoint.")
    return normalized


def discover_allowed_recordings(actor_ids: list[int], split_name: str) -> pd.DataFrame:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"RAVDESS data directory is missing: {DATA_DIR}")

    frames = []
    for actor_id in sorted(actor_ids):
        actor_name = f"Actor_{actor_id:02d}"
        actor_dirs = [
            path for path in DATA_DIR.rglob(actor_name) if path.is_dir()
        ]
        if len(actor_dirs) != 1:
            raise ValueError(
                f"Expected exactly one {actor_name} directory under {DATA_DIR}; "
                f"found {len(actor_dirs)}."
            )
        frame = discover_ravdess(actor_dirs[0])
        if frame.empty:
            raise ValueError(
                f"No recognized speech recordings found for {actor_name} "
                f"in the {split_name} dataset."
            )
        actual_ids = set(frame["actor_id"].astype(int).unique())
        if actual_ids != {actor_id}:
            raise ValueError(
                f"{actor_name} directory contains unexpected parsed actor IDs: "
                f"{sorted(actual_ids)}."
            )
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True).sort_values("path").reset_index(drop=True)
    actual_ids = set(result["actor_id"].astype(int).unique())
    if actual_ids != set(actor_ids):
        raise ValueError(
            f"{split_name} dataset actor IDs {sorted(actual_ids)} do not match "
            f"the split manifest {sorted(actor_ids)}."
        )
    if result["path"].duplicated().any():
        raise ValueError(f"{split_name} dataset contains duplicate recording paths.")
    return result


def load_features(
    frame: pd.DataFrame,
    encoder: LabelEncoder,
    augment: bool,
    dataset_name: str,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    for index, row in enumerate(frame.itertuples(index=False)):
        try:
            audio = load_audio(row.path)
            if augment and rng.random() < NOISE_AUGMENTATION["probability"]:
                snr = float(
                    rng.uniform(
                        NOISE_AUGMENTATION["snr_min_db"],
                        NOISE_AUGMENTATION["snr_max_db"],
                    )
                )
                audio = add_white_noise_at_snr(
                    audio, snr_db=snr, seed=RANDOM_SEED + index
                )
            feature = extract_mfcc_with_deltas(audio)
            label = int(encoder.transform([row.emotion])[0])
            if feature.shape != (MAX_FRAMES, N_MFCC_DELTA, 1):
                raise ValueError(
                    f"Unexpected feature shape {feature.shape}; expected "
                    f"({MAX_FRAMES}, {N_MFCC_DELTA}, 1)."
                )
            if not np.isfinite(feature).all():
                raise ValueError("Feature tensor contains non-finite values.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process required {row.path}: {exc}"
            ) from exc
        features.append(feature)
        labels.append(label)

    if len(features) != len(frame) or len(labels) != len(frame):
        raise RuntimeError(
            f"{dataset_name} feature/label count mismatch: "
            f"{len(frame)} recordings, {len(features)} features, {len(labels)} labels."
        )
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    set_seeds()

    actor_split = load_and_validate_split()
    train_frame = discover_allowed_recordings(actor_split["train"], "training")
    validation_frame = discover_allowed_recordings(
        actor_split["validation"], "validation"
    )
    if set(train_frame["actor_id"]) & set(validation_frame["actor_id"]):
        raise ValueError("Training and validation datasets contain overlapping actors.")
    if (
        set(train_frame["actor_id"]) != set(actor_split["train"])
        or set(validation_frame["actor_id"]) != set(actor_split["validation"])
    ):
        raise ValueError("Dataset actor IDs do not match the actor split manifest.")

    labels = sorted(set(train_frame["emotion"]) | set(validation_frame["emotion"]))
    encoder = LabelEncoder().fit(labels)
    generated_mapping = {
        str(index): str(label) for index, label in enumerate(encoder.classes_)
    }

    existing_mapping_path = MODEL_DIR / "improved_label_mapping.json"
    if not existing_mapping_path.is_file():
        raise FileNotFoundError(
            f"Existing improved-model label mapping is missing: {existing_mapping_path}"
        )
    existing_mapping = json.loads(existing_mapping_path.read_text(encoding="utf-8"))
    if existing_mapping != generated_mapping:
        raise ValueError(
            "Training/validation label mapping does not match "
            "models/improved_label_mapping.json."
        )

    X_train, y_train = load_features(
        train_frame, encoder, augment=True, dataset_name="training", seed_offset=0
    )
    X_validation, y_validation = load_features(
        validation_frame, encoder, augment=False, dataset_name="validation"
    )
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise RuntimeError("Feature and label counts do not match.")
    expected_shape = (MAX_FRAMES, N_MFCC_DELTA, 1)
    if X_train.shape[1:] != expected_shape or X_validation.shape[1:] != expected_shape:
        raise RuntimeError(
            f"Unexpected model input shape: train {X_train.shape[1:]}, "
            f"validation {X_validation.shape[1:]}; expected {expected_shape}."
        )
    expected_classes = np.arange(len(labels))
    if not np.array_equal(np.unique(y_train), expected_classes):
        raise ValueError("Training recordings do not contain every mapped class.")

    class_weights_array = compute_class_weight(
        "balanced", classes=expected_classes, y=y_train
    )
    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(expected_classes, class_weights_array)
    }

    model = build_cnn_lstm(tuple(X_train.shape[1:]), len(labels))
    model.optimizer.learning_rate.assign(5e-4)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=80,
        batch_size=16,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    history_frame = pd.DataFrame(history.history)
    if history_frame.empty or "val_loss" not in history_frame:
        raise RuntimeError("Training did not produce validation-loss history.")
    best_epoch_index = int(history_frame["val_loss"].idxmin())
    best_epoch = best_epoch_index + 1
    best_val_loss = float(history_frame.loc[best_epoch_index, "val_loss"])
    best_val_accuracy = (
        float(history_frame.loc[best_epoch_index, "val_accuracy"])
        if "val_accuracy" in history_frame
        else None
    )

    model.save(MODEL_PATH)
    MAPPING_PATH.write_text(
        json.dumps(generated_mapping, indent=2) + "\n", encoding="utf-8"
    )
    history_frame.to_csv(HISTORY_PATH, index=False)

    epochs = range(1, len(history_frame) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, column, title in (
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ):
        if column in history_frame:
            axis.plot(epochs, history_frame[column], label="Train", linewidth=1.8)
        validation_column = f"val_{column}"
        if validation_column in history_frame:
            axis.plot(
                epochs,
                history_frame[validation_column],
                label="Validation",
                linewidth=1.8,
                linestyle="--",
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(f"Improved CNN-LSTM {title}")
        axis.legend()
        axis.grid(linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(CURVE_PATH, dpi=150)
    plt.close(figure)

    summary = {
        "training_outcome": "completed",
        "actor_split": actor_split,
        "actors_used": {
            "training": sorted(set(train_frame["actor_id"].astype(int))),
            "validation": sorted(set(validation_frame["actor_id"].astype(int))),
        },
        "recording_counts": {
            "training": int(len(train_frame)),
            "validation": int(len(validation_frame)),
        },
        "label_mapping": generated_mapping,
        "feature_configuration": {
            "sample_rate_hz": SAMPLE_RATE,
            "mfcc_coefficients": N_MFCC,
            "delta_orders": [1, 2],
            "feature_coefficients": N_MFCC_DELTA,
            "max_frames": MAX_FRAMES,
            "tensor_shape_per_recording": [
                MAX_FRAMES,
                N_MFCC_DELTA,
                1,
            ],
        },
        "augmentation_configuration": NOISE_AUGMENTATION,
        "training_configuration": {
            "optimizer": "Adam",
            "initial_learning_rate": 5e-4,
            "loss": "sparse_categorical_crossentropy",
            "batch_size": 16,
            "maximum_epochs": 80,
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": 15,
            "restore_best_weights": True,
            "class_weight": "balanced",
            "random_seed": RANDOM_SEED,
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "validation_accuracy_at_best_epoch": best_val_accuracy,
        "epochs_completed": int(len(history_frame)),
        "software_versions": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": package_version("scikit-learn"),
            "librosa": package_version("librosa"),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best validation checkpoint: epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"Experiment artifacts saved under {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
