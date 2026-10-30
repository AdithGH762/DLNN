"""Compare the saved SVM and improved CNN-LSTM on the held-out test manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import MAX_FRAMES, MODEL_DIR, N_MFCC, OUTPUT_DIR, SAMPLE_RATE
from src.dataset import parse_ravdess_filename, validate_test_manifest_labels
from src.metrics import classification_metrics
from src.preprocessing import (
    N_MFCC_DELTA,
    SVM_FEATURE_DIM,
    extract_mfcc_with_deltas,
    extract_svm_features,
    load_audio,
)

SVM_MODEL_PATH = MODEL_DIR / "svm_model.pkl"
SVM_SCALER_PATH = MODEL_DIR / "svm_scaler.pkl"
SVM_MAPPING_PATH = MODEL_DIR / "svm_label_mapping.json"
CNN_MODEL_PATH = (
    SCRIPT_PATH.parent
    / "improved_cnn_lstm_run_01"
    / "improved_cnn_lstm.keras"
)
CNN_MAPPING_PATH = CNN_MODEL_PATH.parent / "label_mapping.json"
MANIFEST_PATH = OUTPUT_DIR / "test_manifest.csv"
SPLIT_PATH = MODEL_DIR / "split_actors.json"
RESULTS_DIR = SCRIPT_PATH.parent / "final_model_comparison"

EXPECTED_TEST_ACTORS = {10, 16, 17, 19, 20}
EXPECTED_ACTOR_COUNTS = {actor_id: 60 for actor_id in EXPECTED_TEST_ACTORS}
CLASS_ORDER = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tensorflow": tf.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for distribution, key in (
        ("scikit-learn", "scikit_learn"),
        ("librosa", "librosa"),
        ("joblib", "joblib"),
    ):
        try:
            versions[key] = version(distribution)
        except PackageNotFoundError:
            versions[key] = "not installed"
    return versions


def read_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required label mapping is missing: {path}")
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or mapping != {
        str(index): label for index, label in enumerate(CLASS_ORDER)
    }:
        raise ValueError(
            f"{path} must map class IDs 0-7 to the documented class order."
        )
    return mapping


def validate_split_and_manifest() -> tuple[dict, pd.DataFrame, list[int]]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Actor split manifest is missing: {SPLIT_PATH}")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if not isinstance(split, dict) or set(split) != {"train", "validation", "test"}:
        raise ValueError("Actor split must contain train, validation, and test lists.")
    actor_sets = {}
    for name in ("train", "validation", "test"):
        actors = split[name]
        if (
            not isinstance(actors, list)
            or any(isinstance(actor, bool) or not isinstance(actor, int) for actor in actors)
            or len(actors) != len(set(actors))
        ):
            raise ValueError(f"Actor split '{name}' must contain unique integer IDs.")
        actor_sets[name] = set(actors)
    if (
        actor_sets["train"] & actor_sets["validation"]
        or actor_sets["train"] & actor_sets["test"]
        or actor_sets["validation"] & actor_sets["test"]
    ):
        raise ValueError("Train, validation, and test actor lists must be disjoint.")
    if actor_sets["test"] != EXPECTED_TEST_ACTORS:
        raise ValueError(
            f"Split test actors {sorted(actor_sets['test'])} do not match "
            f"the required actors {sorted(EXPECTED_TEST_ACTORS)}."
        )

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Test manifest is missing: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    if list(manifest.columns) != ["path", "y_true"]:
        raise ValueError("Test manifest must contain exactly the path and y_true columns.")
    if len(manifest) != 300:
        raise ValueError(f"Expected 300 test manifest rows; found {len(manifest)}.")

    resolved_paths = []
    actor_counts = {actor_id: 0 for actor_id in EXPECTED_TEST_ACTORS}
    for row_number, raw_path in enumerate(manifest["path"], start=2):
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Test manifest row {row_number} has an empty path.")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError(
                f"Test manifest row {row_number} path is not absolute: {raw_path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Test manifest row {row_number} recording is missing: {path}"
            )
        parsed = parse_ravdess_filename(path)
        if parsed is None or parsed["actor_id"] not in EXPECTED_TEST_ACTORS:
            raise ValueError(
                f"Test manifest row {row_number} is not a valid test-actor "
                f"RAVDESS speech recording: {path}"
            )
        actor_counts[parsed["actor_id"]] += 1
        resolved_paths.append(path.resolve())

    unique_paths = [path.as_posix().casefold() for path in resolved_paths]
    if len(set(unique_paths)) != 300:
        raise ValueError("Test manifest paths must be unique.")
    if actor_counts != EXPECTED_ACTOR_COUNTS:
        raise ValueError(
            f"Expected 60 recordings per test actor; observed {actor_counts}."
        )
    found_actors = set(actor_counts)
    if found_actors & (actor_sets["train"] | actor_sets["validation"]):
        raise ValueError("Test manifest contains a training or validation actor.")

    svm_mapping = read_mapping(SVM_MAPPING_PATH)
    cnn_mapping = read_mapping(CNN_MAPPING_PATH)
    if svm_mapping != cnn_mapping:
        raise ValueError("SVM and CNN-LSTM label mappings do not match.")
    svm_ids = validate_test_manifest_labels(
        list(zip(manifest["path"], manifest["y_true"])),
        svm_mapping,
        range(len(svm_mapping)),
        "outputs/test_manifest.csv (SVM)",
    )
    cnn_ids = validate_test_manifest_labels(
        list(zip(manifest["path"], manifest["y_true"])),
        cnn_mapping,
        range(len(cnn_mapping)),
        "outputs/test_manifest.csv (CNN-LSTM)",
    )
    if svm_ids != cnn_ids:
        raise ValueError("Model-specific manifest validation produced different labels.")

    manifest = manifest.copy()
    manifest["path"] = [str(path) for path in resolved_paths]
    return split, manifest, svm_ids


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mapping: dict[str, str],
) -> dict:
    result = classification_metrics(y_true, y_pred, CLASS_ORDER)
    raw_report = result["classification_report"]
    per_emotion = {
        emotion: {
            "precision": float(raw_report[emotion]["precision"]),
            "recall": float(raw_report[emotion]["recall"]),
            "f1": float(raw_report[emotion]["f1-score"]),
            "support": int(raw_report[emotion]["support"]),
        }
        for emotion in CLASS_ORDER
    }
    result["weighted_f1"] = float(raw_report["weighted avg"]["f1-score"])
    result["per_emotion"] = per_emotion
    result["class_order"] = [mapping[str(index)] for index in range(len(mapping))]
    result["correct_count"] = int(np.sum(y_true == y_pred))
    result["total_count"] = int(len(y_true))
    return result


def save_confusion_matrix(matrix: list[list[int]], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(np.asarray(matrix), interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_ORDER)),
        yticks=np.arange(len(CLASS_ORDER)),
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        xlabel="Predicted emotion",
        ylabel="True emotion",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = max((max(row) for row in matrix), default=0) / 2
    for row in range(len(CLASS_ORDER)):
        for column in range(len(CLASS_ORDER)):
            value = matrix[row][column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)

    artifact_paths = {
        "svm_model": SVM_MODEL_PATH,
        "svm_scaler": SVM_SCALER_PATH,
        "svm_label_mapping": SVM_MAPPING_PATH,
        "cnn_lstm_model": CNN_MODEL_PATH,
        "cnn_lstm_label_mapping": CNN_MAPPING_PATH,
        "test_manifest": MANIFEST_PATH,
        "actor_split": SPLIT_PATH,
    }
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} artifact is missing: {path}")

    split, manifest, validated_labels = validate_split_and_manifest()
    svm_mapping = read_mapping(SVM_MAPPING_PATH)
    cnn_mapping = read_mapping(CNN_MAPPING_PATH)

    svm_model = joblib.load(SVM_MODEL_PATH)
    svm_scaler = joblib.load(SVM_SCALER_PATH)
    if not hasattr(svm_model, "classes_"):
        raise ValueError("SVM model does not expose its fitted class IDs.")
    svm_class_ids = np.asarray(svm_model.classes_)
    if not np.array_equal(svm_class_ids, np.arange(len(CLASS_ORDER))):
        raise ValueError(f"Unexpected SVM model class IDs: {svm_class_ids.tolist()}.")
    if getattr(svm_scaler, "n_features_in_", None) != SVM_FEATURE_DIM:
        raise ValueError(
            f"SVM scaler must expect {SVM_FEATURE_DIM} features; found "
            f"{getattr(svm_scaler, 'n_features_in_', None)}."
        )
    if getattr(svm_model, "n_features_in_", None) != SVM_FEATURE_DIM:
        raise ValueError(
            f"SVM model must expect {SVM_FEATURE_DIM} features; found "
            f"{getattr(svm_model, 'n_features_in_', None)}."
        )

    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
    cnn_input_shape = tuple(cnn_model.input_shape)
    expected_cnn_shape = (None, MAX_FRAMES, N_MFCC_DELTA, 1)
    if cnn_input_shape != expected_cnn_shape:
        raise ValueError(
            f"CNN-LSTM input shape {cnn_input_shape} does not match "
            f"{expected_cnn_shape}."
        )
    cnn_output_shape = tuple(cnn_model.output_shape)
    if len(cnn_output_shape) != 2 or cnn_output_shape[-1] != len(CLASS_ORDER):
        raise ValueError(
            f"CNN-LSTM output shape {cnn_output_shape} does not match "
            f"{len(CLASS_ORDER)} emotion classes."
        )

    svm_features = []
    cnn_features = []
    for path in manifest["path"]:
        try:
            audio = load_audio(path, sample_rate=SAMPLE_RATE)
            svm_feature = extract_svm_features(audio, sample_rate=SAMPLE_RATE)
            cnn_feature = extract_mfcc_with_deltas(audio, sample_rate=SAMPLE_RATE)
            if svm_feature.shape != (SVM_FEATURE_DIM,):
                raise ValueError(
                    f"SVM feature shape {svm_feature.shape}, expected "
                    f"({SVM_FEATURE_DIM},)."
                )
            if cnn_feature.shape != (MAX_FRAMES, N_MFCC_DELTA, 1):
                raise ValueError(
                    f"CNN-LSTM feature shape {cnn_feature.shape}, expected "
                    f"({MAX_FRAMES}, {N_MFCC_DELTA}, 1)."
                )
            if not np.isfinite(svm_feature).all() or not np.isfinite(cnn_feature).all():
                raise ValueError("Extracted features contain non-finite values.")
        except Exception as exc:
            raise RuntimeError(f"Failed to process test recording {path}: {exc}") from exc
        svm_features.append(svm_feature)
        cnn_features.append(cnn_feature)

    X_svm = np.stack(svm_features)
    X_cnn = np.stack(cnn_features)
    y_true = np.asarray(validated_labels, dtype=np.int64)
    if len(X_svm) != 300 or len(X_cnn) != 300 or len(y_true) != 300:
        raise RuntimeError("Feature and label arrays must each contain 300 records.")

    try:
        X_svm_scaled = svm_scaler.transform(X_svm)
        svm_predictions = np.asarray(svm_model.predict(X_svm_scaled), dtype=np.int64)
        cnn_probabilities = np.asarray(
            cnn_model.predict(X_cnn, batch_size=32, verbose=0)
        )
    except Exception as exc:
        raise RuntimeError(f"Model inference failed: {exc}") from exc
    if svm_predictions.shape != (300,):
        raise RuntimeError(f"SVM returned unexpected prediction shape {svm_predictions.shape}.")
    if cnn_probabilities.shape != (300, len(CLASS_ORDER)):
        raise RuntimeError(
            f"CNN-LSTM returned unexpected output shape {cnn_probabilities.shape}."
        )
    if (
        np.any(svm_predictions < 0)
        or np.any(svm_predictions >= len(CLASS_ORDER))
        or not np.isfinite(cnn_probabilities).all()
    ):
        raise RuntimeError("Model inference returned invalid class IDs or probabilities.")
    cnn_predictions = np.argmax(cnn_probabilities, axis=1).astype(np.int64)

    svm_metrics = evaluate_predictions(y_true, svm_predictions, svm_mapping)
    cnn_metrics = evaluate_predictions(y_true, cnn_predictions, cnn_mapping)
    run_timestamp = datetime.now(timezone.utc).isoformat()
    provenance = {
        "run_timestamp_utc": run_timestamp,
        "test_manifest_sha256": sha256_file(MANIFEST_PATH),
        "artifact_sha256": {
            name: sha256_file(path) for name, path in artifact_paths.items()
        },
        "actor_split": split,
        "test_actor_counts": {
            str(actor_id): 60 for actor_id in sorted(EXPECTED_TEST_ACTORS)
        },
        "test_record_count": 300,
        "class_order": CLASS_ORDER,
        "software_versions": software_versions(),
        "evaluation": "clean audio only; no augmentation, tuning, or retraining",
    }

    predictions_frame = pd.DataFrame(
        {
            "path": manifest["path"],
            "actor_id": [
                parse_ravdess_filename(Path(path))["actor_id"]
                for path in manifest["path"]
            ],
            "y_true": y_true,
            "true_emotion": [CLASS_ORDER[class_id] for class_id in y_true],
            "svm_y_pred": svm_predictions,
            "svm_emotion": [CLASS_ORDER[class_id] for class_id in svm_predictions],
            "cnn_lstm_y_pred": cnn_predictions,
            "cnn_lstm_emotion": [
                CLASS_ORDER[class_id] for class_id in cnn_predictions
            ],
        }
    )
    final_comparison = {
        "provenance": provenance,
        "models": {
            "svm_baseline": svm_metrics,
            "improved_cnn_lstm": cnn_metrics,
        },
    }
    summary = {
        "provenance": provenance,
        "correct_predictions": {
            "svm_baseline": f"{svm_metrics['correct_count']}/300",
            "improved_cnn_lstm": f"{cnn_metrics['correct_count']}/300",
        },
        "accuracy": {
            "svm_baseline": svm_metrics["accuracy"],
            "improved_cnn_lstm": cnn_metrics["accuracy"],
        },
        "macro_f1": {
            "svm_baseline": svm_metrics["macro_f1"],
            "improved_cnn_lstm": cnn_metrics["macro_f1"],
        },
        "warning": "Final held-out test results must not be used for tuning or checkpoint selection.",
    }

    predictions_frame.to_csv(RESULTS_DIR / "final_predictions.csv", index=False)
    (RESULTS_DIR / "final_comparison.json").write_text(
        json.dumps(final_comparison, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_confusion_matrix(
        svm_metrics["confusion_matrix"],
        "SVM baseline — held-out clean test",
        RESULTS_DIR / "confusion_matrix_svm.png",
    )
    save_confusion_matrix(
        cnn_metrics["confusion_matrix"],
        "Improved CNN-LSTM — held-out clean test",
        RESULTS_DIR / "confusion_matrix_cnn_lstm.png",
    )
    print(f"Final comparison results saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
