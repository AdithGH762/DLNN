"""train_improved.py — Improved CNN-LSTM for EmotionSense.

Improvements over the original train.py:
  1. MFCC + delta + delta-delta features  (200, 120, 1) instead of (200, 40, 1)
  2. Online noise augmentation on training batches  (random SNR 5–30 dB, p=0.5)
  3. Class weighting for imbalanced neutral class
  4. Increased patience (15) and lower initial LR (5e-4) + cosine decay
  5. Saves artifacts to models/improved_* so the original baseline is preserved

Uses the SAME actor split from models/split_actors.json.
Run:   python train_improved.py
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, RANDOM_SEED
from src.dataset import discover_ravdess
from src.preprocessing import load_audio, extract_mfcc_with_deltas
from src.model import build_cnn_lstm
from src.noise import add_white_noise_at_snr

# ── Model save paths (separate from original baseline) ────────────────────────
IMPROVED_MODEL  = MODEL_DIR / "improved_cnn_lstm.keras"
IMPROVED_MAP    = MODEL_DIR / "improved_label_mapping.json"


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split() -> tuple[list[int], list[int], list[int]]:
    split_path = MODEL_DIR / "split_actors.json"
    if not split_path.exists():
        raise SystemExit("models/split_actors.json missing. Run `python train.py` first.")
    s = json.loads(split_path.read_text(encoding="utf-8"))
    return s["train"], s["validation"], s["test"]


def load_features(frame: pd.DataFrame, encoder: LabelEncoder,
                  augment: bool = False, aug_prob: float = 0.5,
                  snr_min: float = 5.0, snr_max: float = 30.0,
                  seed_offset: int = 0) -> tuple[np.ndarray, np.ndarray, list]:
    """Extract delta-MFCC features, optionally adding online noise augmentation."""
    X, y, kept = [], [], []
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    for i, row in enumerate(frame.itertuples(index=False)):
        try:
            audio = load_audio(row.path)
            if augment and rng.random() < aug_prob:
                snr = float(rng.uniform(snr_min, snr_max))
                audio = add_white_noise_at_snr(audio, snr_db=snr, seed=RANDOM_SEED + i)
            feat = extract_mfcc_with_deltas(audio)
            X.append(feat)
            y.append(encoder.transform([row.emotion])[0])
            kept.append(row.path)
        except Exception as exc:
            print(f"  Skipping {Path(row.path).name}: {exc}")
    if not X:
        raise ValueError("No usable features extracted.")
    return np.stack(X), np.asarray(y, dtype=np.int64), kept


def main() -> None:
    set_seeds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_actors, val_actors, test_actors = load_split()
    df = discover_ravdess(DATA_DIR)
    if df.empty:
        raise SystemExit(f"No RAVDESS files found under {DATA_DIR}.")

    labels  = sorted(df.emotion.unique())
    encoder = LabelEncoder().fit(labels)
    print("Emotion labels:", list(encoder.classes_))
    print(f"Train: {len(train_actors)} actors | Val: {len(val_actors)} | Test: {len(test_actors)}")

    train_df = df[df.actor_id.isin(train_actors)]
    val_df   = df[df.actor_id.isin(val_actors)]
    test_df  = df[df.actor_id.isin(test_actors)]

    print("\nExtracting training features (with noise augmentation, p=0.5, SNR 5–30 dB)…")
    X_train, y_train, _ = load_features(train_df, encoder, augment=True, aug_prob=0.5,
                                         snr_min=5.0, snr_max=30.0, seed_offset=0)
    print("Extracting validation features (clean)…")
    X_val,   y_val,   _ = load_features(val_df,   encoder, augment=False)
    print("Extracting test features (clean)…")
    X_test,  y_test, test_paths = load_features(test_df, encoder, augment=False)

    print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    # ── Class weights (handles neutral having fewer samples) ───────────────
    cw_vals = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw_vals)}
    print("Class weights:", {labels[i]: round(class_weight[i], 2) for i in range(len(labels))})

    # ── Model ──────────────────────────────────────────────────────────────
    input_shape = tuple(X_train.shape[1:])  # (200, 120, 1)
    model = build_cnn_lstm(input_shape, len(labels))
    model.optimizer.learning_rate.assign(5e-4)  # lower initial LR
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(IMPROVED_MODEL),
            monitor="val_loss", mode="min", save_best_only=True, verbose=0
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=16,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    hist_df = pd.DataFrame(history.history)
    selected_epoch_index = int(hist_df["val_loss"].idxmin())
    selected_epoch = selected_epoch_index + 1
    selected_val_loss = float(hist_df.loc[selected_epoch_index, "val_loss"])

    # EarlyStopping restores the same minimum-val_loss weights saved by ModelCheckpoint.
    model.save(IMPROVED_MODEL)
    with open(IMPROVED_MAP, "w", encoding="utf-8") as f:
        json.dump({str(i): str(lbl) for i, lbl in enumerate(encoder.classes_)}, f, indent=2)

    hist_df.to_csv(OUTPUT_DIR / "improved_training_history.csv", index=False)

    # Save test predictions for evaluation pipeline
    test_probs = model.predict(X_test, batch_size=32, verbose=0)
    test_preds = test_probs.argmax(axis=1)
    pd.DataFrame({
        "path":   test_paths,
        "y_true": y_test.tolist(),
        "y_pred": test_preds.tolist(),
    }).to_csv(OUTPUT_DIR / "improved_test_predictions.csv", index=False)
    pd.DataFrame({"path": test_paths, "y_true": y_test}).to_csv(
        OUTPUT_DIR / "improved_test_manifest.csv", index=False
    )

    # ── Learning curves ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist_df) + 1)
    for ax, col, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        if col in hist_df.columns:
            ax.plot(epochs, hist_df[col], label="Train", linewidth=1.8)
        if f"val_{col}" in hist_df.columns:
            ax.plot(epochs, hist_df[f"val_{col}"], label="Val", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label)
        ax.set_title(f"Improved CNN-LSTM {label}")
        ax.legend(); ax.grid(linestyle="--", alpha=0.4)
    fig.suptitle("EmotionSense Improved — Learning Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "improved_learning_curves.png", dpi=150)
    plt.close(fig)

    print(
        f"\nSelected checkpoint by val_loss: epoch {selected_epoch} "
        f"(val_loss={selected_val_loss:.4f})"
    )
    print(f"Model saved to {IMPROVED_MODEL}")
    print("Run `python evaluate_all.py` to compare all models.")


if __name__ == "__main__":
    main()
