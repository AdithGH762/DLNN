from __future__ import annotations

import hashlib
import json
import pickle
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import EMOTION_CODES, MODEL_DIR, OUTPUT_DIR
from src.dataset import parse_ravdess_filename, validate_test_manifest_labels
from src.metrics import classification_metrics
from src.preprocessing import SVM_FEATURE_DIM, extract_svm_features, load_audio


EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "svm_rbf_tuning"
FINAL_TEST_DIR = EXPERIMENT_DIR / "final_test"
EXPECTED_TEST_ACTORS = {10, 16, 17, 19, 20}
MODELS = {
    "baseline": {
        "model": MODEL_DIR / "svm_model.pkl",
        "scaler": MODEL_DIR / "svm_scaler.pkl",
        "mapping": MODEL_DIR / "svm_label_mapping.json",
    },
    "tuned": {
        "model": EXPERIMENT_DIR / "selected_svm_model.pkl",
        "scaler": EXPERIMENT_DIR / "training_only_scaler.pkl",
        "mapping": EXPERIMENT_DIR / "label_mapping.json",
    },
}
TEST_MANIFEST = OUTPUT_DIR / "test_manifest.csv"
ACTOR_SPLIT = MODEL_DIR / "split_actors.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model_artifacts(model_key: str) -> tuple[object, object, dict[str, str]]:
    paths = MODELS[model_key]
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required {model_key} artifact is missing: {path}")
    try:
        with paths["model"].open("rb") as model_file:
            model = pickle.load(model_file)
        with paths["scaler"].open("rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
    except (OSError, pickle.UnpicklingError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load {model_key} artifacts: {exc}") from exc

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{model_key} label mapping must be a non-empty object.")
    if (
        set(mapping) != {str(index) for index in range(len(mapping))}
        or set(mapping.values()) != set(EMOTION_CODES.values())
        or len(set(mapping.values())) != len(mapping)
    ):
        raise ValueError(f"{model_key} label mapping is invalid.")
    if not hasattr(model, "classes_"):
        raise ValueError(f"{model_key} model does not expose class IDs.")
    model_classes = np.asarray(model.classes_)
    try:
        normalized_classes = [int(value) for value in model_classes]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{model_key} model has invalid class IDs.") from exc
    expected_classes = list(range(len(mapping)))
    if normalized_classes != expected_classes:
        raise ValueError(
            f"{model_key} model class IDs do not match its label mapping."
        )
    scaler_features = getattr(scaler, "n_features_in_", None)
    if scaler_features != SVM_FEATURE_DIM:
        raise ValueError(
            f"{model_key} scaler expects {scaler_features!r} features; "
            f"the project SVM pipeline produces {SVM_FEATURE_DIM}."
        )
    return model, scaler, mapping


def _load_test_manifest(
    mapping: dict[str, str], class_ids: list[int]
) -> tuple[pd.DataFrame, dict[int, int]]:
    if not TEST_MANIFEST.is_file():
        raise FileNotFoundError(f"Test manifest is missing: {TEST_MANIFEST}")
    if not ACTOR_SPLIT.is_file():
        raise FileNotFoundError(f"Actor split is missing: {ACTOR_SPLIT}")

    try:
        manifest = pd.read_csv(TEST_MANIFEST)
        split = json.loads(ACTOR_SPLIT.read_text(encoding="utf-8"))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read test manifest or actor split: {exc}") from exc
    if (
        manifest.empty
        or len(manifest) != 300
        or not {"path", "y_true"}.issubset(manifest.columns)
    ):
        raise ValueError(
            "The test manifest must contain exactly 300 rows and path/y_true columns."
        )
    if manifest["path"].isna().any() or manifest["path"].duplicated().any():
        raise ValueError("The test manifest has an empty or duplicate recording path.")

    try:
        split_train_actors = set(map(int, split["train"]))
        split_validation_actors = set(map(int, split["validation"]))
        split_test_actors = set(map(int, split["test"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The actor split has an invalid actor list.") from exc
    if (
        not split_train_actors
        or not split_validation_actors
        or not split_test_actors
        or split_train_actors & split_validation_actors
        or split_train_actors & split_test_actors
        or split_validation_actors & split_test_actors
    ):
        raise ValueError("The saved actor split is empty or contains overlapping actors.")
    if split_test_actors != EXPECTED_TEST_ACTORS:
        raise ValueError(
            "The saved test split does not match expected actors "
            f"{sorted(EXPECTED_TEST_ACTORS)}."
        )
    actor_counts: dict[int, int] = {}
    for raw_path in manifest["path"]:
        recording_path = Path(str(raw_path))
        if not recording_path.is_absolute() or not recording_path.is_file():
            raise ValueError(
                f"Manifest path must be an existing absolute path: {raw_path}"
            )
        parsed = parse_ravdess_filename(recording_path)
        if parsed is None or parsed["actor_id"] not in EXPECTED_TEST_ACTORS:
            raise ValueError(
                f"Manifest contains an invalid recording or unexpected actor: {raw_path}"
            )
        actor_id = int(parsed["actor_id"])
        actor_counts[actor_id] = actor_counts.get(actor_id, 0) + 1
    if actor_counts != {actor_id: 60 for actor_id in EXPECTED_TEST_ACTORS}:
        raise ValueError(f"Unexpected held-out actor counts: {actor_counts}")

    validated_labels = validate_test_manifest_labels(
        manifest[["path", "y_true"]].itertuples(index=False, name=None),
        mapping,
        class_ids,
        TEST_MANIFEST.name,
    )
    manifest = manifest[["path"]].copy()
    manifest["y_true"] = validated_labels
    manifest["path"] = manifest["path"].map(
        lambda value: str(Path(str(value)).resolve())
    )
    if manifest["path"].duplicated().any():
        raise ValueError("Resolved test manifest paths are not unique.")
    label_counts = pd.Series(validated_labels).value_counts().to_dict()
    expected_label_counts = {
        int(class_id): (20 if emotion == "neutral" else 40)
        for class_id, emotion in mapping.items()
    }
    if label_counts != expected_label_counts:
        raise ValueError(
            f"Unexpected test label counts: {label_counts}; "
            f"expected {expected_label_counts}."
        )
    return manifest, actor_counts


def main() -> None:
    if FINAL_TEST_DIR.exists():
        raise FileExistsError(
            f"Final-test output already exists; refusing to overwrite: {FINAL_TEST_DIR}"
        )

    models = {
        model_key: _load_model_artifacts(model_key)
        for model_key in ("baseline", "tuned")
    }
    baseline_mapping = models["baseline"][2]
    tuned_mapping = models["tuned"][2]
    if baseline_mapping != tuned_mapping:
        raise ValueError("Baseline and tuned model label mappings do not match.")

    class_ids = list(range(len(baseline_mapping)))
    labels = [baseline_mapping[str(class_id)] for class_id in class_ids]
    manifest, actor_counts = _load_test_manifest(baseline_mapping, class_ids)

    features: list[np.ndarray] = []
    recording_hashes: list[str] = []
    for row in manifest.itertuples(index=False):
        recording_path = Path(row.path)
        try:
            recording_hashes.append(_sha256_file(recording_path))
            audio = load_audio(recording_path)
            feature = extract_svm_features(audio)
            if (
                feature.shape != (SVM_FEATURE_DIM,)
                or not np.isfinite(feature).all()
            ):
                raise ValueError(
                    f"Expected {SVM_FEATURE_DIM} finite SVM features; "
                    f"received shape {feature.shape}."
                )
            features.append(feature)
        except Exception as exc:
            raise RuntimeError(
                f"Could not process required test recording {recording_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    feature_matrix = np.stack(features).astype(np.float32)
    if len(feature_matrix) != 300:
        raise RuntimeError(
            f"Processed {len(feature_matrix)} test recordings; expected exactly 300."
        )

    evidence_rows = []
    metric_results = {}
    for model_key, (model, scaler, mapping) in models.items():
        try:
            scaled_features = scaler.transform(feature_matrix)
            predictions = np.asarray(model.predict(scaled_features), dtype=np.int64)
        except Exception as exc:
            raise RuntimeError(
                f"{model_key} prediction failed on the complete test cohort: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if predictions.shape != (300,):
            raise RuntimeError(
                f"{model_key} returned {predictions.shape!r}; expected 300 predictions."
            )
        if not set(np.unique(predictions)).issubset(set(class_ids)):
            raise RuntimeError(f"{model_key} returned an unknown class ID.")

        y_true = manifest["y_true"].to_numpy(dtype=np.int64)
        result = classification_metrics(y_true, predictions, labels)
        report = result["classification_report"]
        metric_results[model_key] = {
            "accuracy": result["accuracy"],
            "macro_precision": result["macro_precision"],
            "macro_recall": result["macro_recall"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_emotion": {
                emotion: {
                    "precision": float(report[emotion]["precision"]),
                    "recall": float(report[emotion]["recall"]),
                    "f1": float(report[emotion]["f1-score"]),
                    "support": int(report[emotion]["support"]),
                }
                for emotion in labels
            },
            "confusion_matrix": {
                "labels_order": labels,
                "rows_true_columns_predicted": result["confusion_matrix"],
            },
        }
        for index, row in enumerate(manifest.itertuples(index=False)):
            true_id = int(row.y_true)
            predicted_id = int(predictions[index])
            parsed = parse_ravdess_filename(Path(row.path))
            if parsed is None:
                raise RuntimeError(
                    f"Validated test recording became invalid: {row.path}"
                )
            evidence_rows.append({
                "model_key": model_key,
                "path": row.path,
                "recording_sha256": recording_hashes[index],
                "actor_id": int(parsed["actor_id"]),
                "y_true": true_id,
                "true_label": mapping[str(true_id)],
                "y_pred": predicted_id,
                "predicted_label": mapping[str(predicted_id)],
            })

    provenance_inputs = {
        "baseline_model": MODELS["baseline"]["model"],
        "baseline_scaler": MODELS["baseline"]["scaler"],
        "baseline_label_mapping": MODELS["baseline"]["mapping"],
        "tuned_model": MODELS["tuned"]["model"],
        "tuned_scaler": MODELS["tuned"]["scaler"],
        "tuned_label_mapping": MODELS["tuned"]["mapping"],
        "test_manifest": TEST_MANIFEST,
        "actor_split": ACTOR_SPLIT,
        "tuning_experiment_summary": EXPERIMENT_DIR / "experiment_summary.json",
        "comparison_script": Path(__file__).resolve(),
    }
    provenance = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "baseline_vs_validation_selected_svm_clean_test",
        "test_set_used_for_selection": False,
        "test_manifest_rows": len(manifest),
        "test_actor_counts": {
            str(actor_id): count for actor_id, count in sorted(actor_counts.items())
        },
        "class_order": labels,
        "features": "40 MFCCs pooled by mean, std, min, max (160 dimensions)",
        "preprocessing": "project load_audio mono at 16 kHz and extract_svm_features",
        "scalers": {
            "baseline": "models/svm_scaler.pkl",
            "tuned": "experiments/svm_rbf_tuning/training_only_scaler.pkl",
        },
        "model_artifacts": {
            key: {
                artifact_name: str(path.relative_to(PROJECT_ROOT).as_posix())
                for artifact_name, path in paths.items()
            }
            for key, paths in MODELS.items()
        },
        "inputs_sha256": {
            name: _sha256_file(path) for name, path in provenance_inputs.items()
        },
        "recording_sha256_by_path": {
            row.path: recording_hashes[index]
            for index, row in enumerate(manifest.itertuples(index=False))
        },
        "metrics_file": "metrics.json",
        "predictions_file": "predictions.csv",
    }

    # Create output only after all inputs and predictions have passed validation.
    FINAL_TEST_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(evidence_rows).to_csv(
        FINAL_TEST_DIR / "predictions.csv", index=False
    )
    (FINAL_TEST_DIR / "metrics.json").write_text(
        json.dumps(metric_results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (FINAL_TEST_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Final clean-test comparison saved to {FINAL_TEST_DIR}")
    print(json.dumps(metric_results, indent=2))


if __name__ == "__main__":
    main()
