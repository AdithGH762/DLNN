from __future__ import annotations
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive — no GUI window during training
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, SAMPLE_RATE, RANDOM_SEED
from src.dataset import discover_ravdess
from src.preprocessing import load_audio, extract_mfcc
from src.model import build_cnn_lstm


def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def actor_splits(df, seed=RANDOM_SEED):
    """Create actor-independent train/validation/test partitions."""
    actors = np.array(sorted(df.actor_id.unique()))
    if len(actors) < 5:
        raise ValueError("At least 5 distinct actors are needed for an actor-independent split.")
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    n_test = max(1, round(len(actors) * 0.2))
    n_val = max(1, round(len(actors) * 0.2))
    test_actors = set(actors[:n_test])
    val_actors = set(actors[n_test:n_test+n_val])
    train_actors = set(actors[n_test+n_val:])
    if not train_actors:
        raise ValueError("Not enough actors left for training.")
    train = df[df.actor_id.isin(train_actors)].copy()
    val = df[df.actor_id.isin(val_actors)].copy()
    test = df[df.actor_id.isin(test_actors)].copy()
    return train, val, test


def load_features(frame, label_encoder):
    X, y, kept = [], [], []
    for row in frame.itertuples(index=False):
        try:
            audio = load_audio(row.path)
            feature = extract_mfcc(audio)
            X.append(feature)
            y.append(label_encoder.transform([row.emotion])[0])
            kept.append(row.path)
        except Exception as exc:
            print(f"Skipping unreadable file {row.path}: {exc}")
    if not X:
        raise ValueError("No usable audio features could be extracted.")
    return np.stack(X), np.asarray(y, dtype=np.int64), kept


def main():
    set_seeds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = discover_ravdess(DATA_DIR)
    if df.empty:
        raise SystemExit(
            f"No RAVDESS files found in {DATA_DIR}. See data/README.md, extract the speech dataset, then rerun."
        )
    labels = sorted(df.emotion.unique())
    encoder = LabelEncoder().fit(labels)
    train_df, val_df, test_df = actor_splits(df)
    split_actors = {
        "train": sorted(map(int, train_df.actor_id.unique())),
        "validation": sorted(map(int, val_df.actor_id.unique())),
        "test": sorted(map(int, test_df.actor_id.unique())),
    }
    print("Actor split:", json.dumps(split_actors, indent=2))
    print("Emotion labels actually found:", list(encoder.classes_))
    X_train, y_train, _ = load_features(train_df, encoder)
    X_val, y_val, _ = load_features(val_df, encoder)
    X_test, y_test, test_paths = load_features(test_df, encoder)

    model = build_cnn_lstm(tuple(X_train.shape[1:]), len(encoder.classes_))
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "emotionsense_cnn_lstm.keras"),
            monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(MODEL_DIR / "emotionsense_cnn_lstm.keras")
    with open(MODEL_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(i): str(label) for i, label in enumerate(encoder.classes_)}, f, indent=2)
    with open(MODEL_DIR / "split_actors.json", "w", encoding="utf-8") as f:
        json.dump(split_actors, f, indent=2)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    pd.DataFrame({"path": test_paths, "y_true": y_test}).to_csv(OUTPUT_DIR / "test_manifest.csv", index=False)

    # Quick learning curve saved immediately after training
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist_df) + 1)
    for ax, col, label in [
        (axes[0], "loss",     "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        if col in hist_df.columns:
            ax.plot(epochs, hist_df[col], label="Train", linewidth=1.8)
        val_col = f"val_{col}"
        if val_col in hist_df.columns:
            ax.plot(epochs, hist_df[val_col], label="Val", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Training {label}")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)
    fig.suptitle("EmotionSense — Learning Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)

    print(f"Saved model to {MODEL_DIR / 'emotionsense_cnn_lstm.keras'}")
    print(f"Learning curves saved to {OUTPUT_DIR / 'learning_curves.png'}")
    print("Run `python evaluate.py` to evaluate clean and noisy test audio.")


if __name__ == "__main__":
    main()
