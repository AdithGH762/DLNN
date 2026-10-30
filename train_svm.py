"""train_svm.py — MFCC + SVM baseline for EmotionSense.

Uses the SAME actor split saved by train.py (models/split_actors.json).
Fits StandardScaler on training features only.
Saves model artifacts to models/svm_*.

Run:   python train_svm.py
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, RANDOM_SEED
from src.dataset import discover_ravdess
from src.preprocessing import load_audio, extract_svm_features


def load_split() -> tuple[list[int], list[int], list[int]]:
    split_path = MODEL_DIR / "split_actors.json"
    if not split_path.exists():
        raise SystemExit("models/split_actors.json missing. Run `python train.py` first.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return split["train"], split["validation"], split["test"]


def build_xy(
    df: pd.DataFrame, encoder: LabelEncoder
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, paths = [], [], []
    for row in df.itertuples(index=False):
        try:
            audio = load_audio(row.path)
            feat  = extract_svm_features(audio)
            X.append(feat)
            y.append(encoder.transform([row.emotion])[0])
            paths.append(row.path)
        except Exception as exc:
            print(f"  Skipping {Path(row.path).name}: {exc}")
    if not X:
        raise ValueError("No usable features extracted.")
    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        paths,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Reuse the same actor split ─────────────────────────────────────────
    train_actors, val_actors, test_actors = load_split()
    df = discover_ravdess(DATA_DIR)
    if df.empty:
        raise SystemExit(f"No RAVDESS files found under {DATA_DIR}.")

    labels  = sorted(df.emotion.unique())
    encoder = LabelEncoder().fit(labels)

    train_df = df[df.actor_id.isin(train_actors)]
    val_df   = df[df.actor_id.isin(val_actors)]
    test_df  = df[df.actor_id.isin(test_actors)]

    print(f"Train actors: {sorted(train_actors)}  ({len(train_df)} files)")
    print(f"Val   actors: {sorted(val_actors)}  ({len(val_df)} files)")
    print(f"Test  actors: {sorted(test_actors)}  ({len(test_df)} files)")

    print("\nExtracting training features…")
    X_train, y_train, _ = build_xy(train_df, encoder)
    print("Extracting validation features…")
    X_val,   y_val,   _ = build_xy(val_df,   encoder)
    print("Extracting test features…")
    X_test,  y_test, test_paths = build_xy(test_df, encoder)

    # ── Fit scaler on training data ONLY ──────────────────────────────────
    scaler  = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── Grid search over C (using validation set — no test leakage) ───────
    best_c, best_val_acc = 1.0, 0.0
    for C in [0.1, 1.0, 10.0, 100.0]:
        svm = SVC(C=C, kernel="rbf", class_weight="balanced",
                  random_state=RANDOM_SEED, probability=True)
        svm.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, svm.predict(X_val))
        print(f"  C={C:5.1f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_c       = C
            best_svm     = svm

    print(f"\nBest C={best_c}  val_acc={best_val_acc:.4f}")

    # ── Final evaluation on held-out test set ─────────────────────────────
    test_acc = accuracy_score(y_test, best_svm.predict(X_test))
    print(f"Test accuracy: {test_acc:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────
    with open(MODEL_DIR / "svm_model.pkl", "wb") as f:
        pickle.dump(best_svm, f)
    with open(MODEL_DIR / "svm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "svm_label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(i): str(lbl) for i, lbl in enumerate(encoder.classes_)}, f, indent=2)
    with open(MODEL_DIR / "svm_config.json", "w", encoding="utf-8") as f:
        json.dump({"best_C": best_c, "val_accuracy": best_val_acc,
                   "test_accuracy": test_acc, "kernel": "rbf",
                   "class_weight": "balanced", "feature": "mfcc_mean_std_min_max",
                   "feature_dim": int(X_train.shape[1])}, f, indent=2)

    # Save test predictions for evaluation pipeline
    preds = best_svm.predict(X_test)
    proba = best_svm.predict_proba(X_test)
    pd.DataFrame({
        "path":   test_paths,
        "y_true": y_test.tolist(),
        "y_pred": preds.tolist(),
    }).to_csv(OUTPUT_DIR / "svm_test_predictions.csv", index=False)

    print(f"\nSVM artifacts saved to {MODEL_DIR}")
    print("Run `python evaluate_all.py` to compare all models.")


if __name__ == "__main__":
    main()
