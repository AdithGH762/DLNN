"""Train the missing original CNN-LSTM without touching the held-out test set."""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from config import DATA_DIR, MAX_FRAMES, MODEL_DIR, N_MFCC, RANDOM_SEED
from src.dataset import discover_ravdess
from src.model import build_cnn_lstm
from src.preprocessing import extract_mfcc, load_audio

OUTPUT_MODEL = MODEL_DIR / "emotionsense_cnn_lstm.keras"
MAPPING_PATH = MODEL_DIR / "label_mapping.json"


def load_actor_split() -> dict[str, list[int]]:
    split_path = MODEL_DIR / "split_actors.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_counts = {"train": 14, "validation": 5, "test": 5}
    if not isinstance(split, dict) or set(split) != set(expected_counts):
        raise ValueError("Actor split must contain train, validation, and test.")
    normalized = {}
    for name, count in expected_counts.items():
        actors = split[name]
        if (
            not isinstance(actors, list)
            or any(isinstance(actor, bool) or not isinstance(actor, int) for actor in actors)
            or len(actors) != count
            or len(set(actors)) != count
            or any(actor < 1 or actor > 24 for actor in actors)
        ):
            raise ValueError(f"Actor split '{name}' must contain {count} unique actors.")
        normalized[name] = actors
    partitions = [set(normalized[name]) for name in expected_counts]
    if any(
        partitions[left] & partitions[right]
        for left in range(len(partitions))
        for right in range(left + 1, len(partitions))
    ):
        raise ValueError("Train, validation, and test actor lists must be disjoint.")
    return normalized


def recordings_for_actors(actor_ids: list[int], split_name: str) -> pd.DataFrame:
    frames = []
    for actor_id in sorted(actor_ids):
        actor_name = f"Actor_{actor_id:02d}"
        directories = [
            path for path in DATA_DIR.rglob(actor_name)
            if path.is_dir() and path.name == actor_name
        ]
        if len(directories) != 1:
            raise ValueError(
                f"Expected exactly one {actor_name} directory; found {len(directories)}."
            )
        frame = discover_ravdess(directories[0])
        if frame.empty or set(frame["actor_id"].astype(int)) != {actor_id}:
            raise ValueError(
                f"{actor_name} has no valid speech recordings or contains mismatched actors."
            )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values("path").reset_index(drop=True)
    if result["path"].duplicated().any():
        raise ValueError(f"{split_name} contains duplicate recording paths.")
    if set(result["actor_id"].astype(int)) != set(actor_ids):
        raise ValueError(f"{split_name} actor IDs differ from the saved actor split.")
    return result


def load_features(
    frame: pd.DataFrame,
    emotion_to_id: dict[str, int],
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row in frame.itertuples(index=False):
        try:
            audio = load_audio(row.path)
            feature = extract_mfcc(audio)
            if feature.shape != (MAX_FRAMES, N_MFCC, 1):
                raise ValueError(f"Unexpected MFCC tensor shape: {feature.shape}.")
            if not np.isfinite(feature).all():
                raise ValueError("MFCC tensor contains non-finite values.")
            label = emotion_to_id[row.emotion]
        except Exception as exc:
            raise RuntimeError(f"Failed to process required recording {row.path}: {exc}") from exc
        features.append(feature)
        labels.append(label)
    if len(features) != len(frame) or len(labels) != len(frame):
        raise RuntimeError(
            f"{dataset_name} has {len(frame)} recordings, {len(features)} features, "
            f"and {len(labels)} labels."
        )
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def main() -> None:
    if OUTPUT_MODEL.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {OUTPUT_MODEL}")
    if not MAPPING_PATH.is_file():
        raise FileNotFoundError(f"Original model label mapping is missing: {MAPPING_PATH}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MAPPING_PATH.open(encoding="utf-8") as mapping_file:
        mapping = json.load(mapping_file)
    expected_mapping = {
        str(index): emotion
        for index, emotion in enumerate(sorted(set(mapping.values())))
    }
    if mapping != expected_mapping or len(mapping) != 8:
        raise ValueError("Original model label mapping must be the eight sorted RAVDESS classes.")
    emotion_to_id = {emotion: int(class_id) for class_id, emotion in mapping.items()}

    split = load_actor_split()
    train_frame = recordings_for_actors(split["train"], "training")
    validation_frame = recordings_for_actors(split["validation"], "validation")
    if set(train_frame["actor_id"]) & set(validation_frame["actor_id"]):
        raise ValueError("Training and validation actor sets overlap.")
    if set(train_frame["emotion"]) | set(validation_frame["emotion"]) != set(emotion_to_id):
        raise ValueError("Training and validation data do not cover the saved label mapping.")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    x_train, y_train = load_features(train_frame, emotion_to_id, "training")
    x_validation, y_validation = load_features(
        validation_frame, emotion_to_id, "validation"
    )
    model = build_cnn_lstm(tuple(x_train.shape[1:]), len(mapping))

    with tempfile.TemporaryDirectory(
        prefix=".original_cnn_lstm_training_",
        dir=MODEL_DIR,
    ) as staging_dir:
        staging_model = Path(staging_dir) / OUTPUT_MODEL.name
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(staging_model), monitor="val_loss", save_best_only=True
            ),
        ]
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=60,
            batch_size=16,
            callbacks=callbacks,
            verbose=1,
        )
        model.save(staging_model)
        if OUTPUT_MODEL.exists():
            raise FileExistsError(f"Refusing to overwrite existing model: {OUTPUT_MODEL}")
        staging_model.rename(OUTPUT_MODEL)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"Saved original CNN-LSTM model to {OUTPUT_MODEL}")
    print(
        f"Best validation-loss epoch: {best_epoch}; "
        f"validation accuracy: {history.history['val_accuracy'][best_epoch - 1]:.4f}"
    )
    print(
        "Training used only the saved training and validation actors; "
        "no test recordings were loaded or evaluated."
    )


if __name__ == "__main__":
    main()
