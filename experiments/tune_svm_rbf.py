from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import DATA_DIR, EMOTION_CODES, MODEL_DIR, RANDOM_SEED
from src.dataset import discover_ravdess
from src.preprocessing import SVM_FEATURE_DIM, extract_svm_features, load_audio


EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "svm_rbf_tuning"
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMA_VALUES = ("scale", 0.001, 0.01, 0.1)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict]:
    split_path = MODEL_DIR / "split_actors.json"
    mapping_path = MODEL_DIR / "svm_label_mapping.json"
    config_path = MODEL_DIR / "svm_config.json"
    for path in (split_path, mapping_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input is missing: {path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    try:
        train_actors = set(map(int, split["train"]))
        validation_actors = set(map(int, split["validation"]))
        test_actors = set(map(int, split["test"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The saved actor split is malformed.") from exc
    if not train_actors or not validation_actors or not test_actors:
        raise ValueError("The saved actor split must have non-empty partitions.")
    if (
        train_actors & validation_actors
        or train_actors & test_actors
        or validation_actors & test_actors
    ):
        raise ValueError("The saved actor split contains overlapping actor IDs.")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("The saved SVM label mapping is malformed.")
    expected_ids = {str(index) for index in range(len(mapping))}
    if (
        set(mapping) != expected_ids
        or len(set(mapping.values())) != len(mapping)
        or set(mapping.values()) != set(EMOTION_CODES.values())
    ):
        raise ValueError("The saved SVM label mapping is malformed.")
    emotion_to_id = {
        emotion: int(class_id) for class_id, emotion in mapping.items()
    }

    existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        existing_config.get("kernel") != "rbf"
        or existing_config.get("class_weight") != "balanced"
        or existing_config.get("feature") != "mfcc_mean_std_min_max"
        or existing_config.get("feature_dim") != SVM_FEATURE_DIM
    ):
        raise ValueError(
            "The saved SVM configuration does not match the experiment baseline."
        )
    baseline_c = float(existing_config["best_C"])
    baseline_gamma = existing_config.get("gamma", "scale")
    if baseline_c not in C_VALUES or baseline_gamma not in GAMMA_VALUES:
        raise ValueError("The existing SVM configuration is outside the experiment grid.")

    dataset = discover_ravdess(DATA_DIR)
    if dataset.empty:
        raise ValueError(f"No RAVDESS speech recordings were discovered under {DATA_DIR}.")
    train_frame = dataset[dataset["actor_id"].isin(train_actors)].copy()
    validation_frame = dataset[dataset["actor_id"].isin(validation_actors)].copy()
    if train_frame.empty or validation_frame.empty:
        raise ValueError("The saved split has no discovered train or validation recordings.")
    return train_frame, validation_frame, emotion_to_id, existing_config


def _build_xy(
    frame: pd.DataFrame, emotion_to_id: dict[str, int], split_name: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    skipped: list[dict[str, str]] = []

    for row in frame.itertuples(index=False):
        if row.emotion not in emotion_to_id:
            raise ValueError(
                f"Emotion {row.emotion!r} is absent from the saved SVM mapping."
            )
        try:
            audio = load_audio(row.path)
            feature = extract_svm_features(audio)
        except Exception as exc:
            skipped.append({
                "split": split_name,
                "path": str(row.path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        features.append(feature)
        labels.append(emotion_to_id[row.emotion])
        paths.append(str(row.path))

    if not features:
        raise ValueError(f"No usable {split_name} audio features were extracted.")
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        paths,
        skipped,
    )


def main() -> None:
    if EXPERIMENT_DIR.exists():
        raise FileExistsError(
            f"Experiment output already exists; refusing to overwrite: {EXPERIMENT_DIR}"
        )

    train_frame, validation_frame, emotion_to_id, existing_config = _load_inputs()
    X_train, y_train, train_paths, skipped_train = _build_xy(
        train_frame, emotion_to_id, "train"
    )
    X_validation, y_validation, validation_paths, skipped_validation = _build_xy(
        validation_frame, emotion_to_id, "validation"
    )

    expected_class_ids = set(emotion_to_id.values())
    if set(np.unique(y_train)) != expected_class_ids:
        raise ValueError("Training features do not contain every mapped emotion class.")
    if set(np.unique(y_validation)) != expected_class_ids:
        raise ValueError(
            "Validation features do not contain every mapped emotion class."
        )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    class_ids = sorted(expected_class_ids)

    baseline_c = float(existing_config["best_C"])
    baseline_gamma = existing_config.get("gamma", "scale")
    result_rows = []
    best_model = None
    best_selection_key = None
    for c_value in C_VALUES:
        for gamma_value in GAMMA_VALUES:
            model = SVC(
                C=c_value,
                gamma=gamma_value,
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                probability=True,
            )
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_validation_scaled)
            validation_accuracy = float(
                accuracy_score(y_validation, predictions)
            )
            validation_macro_f1 = float(f1_score(
                y_validation,
                predictions,
                labels=class_ids,
                average="macro",
                zero_division=0,
            ))
            is_existing_configuration = (
                c_value == baseline_c and gamma_value == baseline_gamma
            )
            result_row = {
                "C": c_value,
                "gamma": str(gamma_value),
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
                "is_existing_configuration": is_existing_configuration,
            }
            result_rows.append(result_row)
            selection_key = (
                validation_macro_f1,
                validation_accuracy,
                is_existing_configuration,
                -len(result_rows),
            )
            if best_selection_key is None or selection_key > best_selection_key:
                best_selection_key = selection_key
                best_model = model
                best_row = result_row

    baseline_row = next(
        row for row in result_rows if row["is_existing_configuration"]
    )
    if best_model is None:
        raise RuntimeError("The SVM hyperparameter grid produced no model.")

    # Create the output directory exclusively, so a concurrent run cannot overwrite it.
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(result_rows).to_csv(
        EXPERIMENT_DIR / "validation_grid_results.csv", index=False
    )
    pd.DataFrame(
        skipped_train + skipped_validation,
        columns=["split", "path", "error"],
    ).to_csv(EXPERIMENT_DIR / "skipped_records.csv", index=False)
    with (EXPERIMENT_DIR / "selected_svm_model.pkl").open("xb") as model_file:
        pickle.dump(best_model, model_file)
    with (EXPERIMENT_DIR / "training_only_scaler.pkl").open("xb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    (EXPERIMENT_DIR / "label_mapping.json").write_text(
        json.dumps(
            {str(class_id): emotion for emotion, class_id in emotion_to_id.items()},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    summary = {
        "experiment": "isolated_svm_rbf_hyperparameter_tuning",
        "selection_metric": "validation_macro_f1",
        "tie_breaker": "validation_accuracy",
        "selected_configuration": {
            "C": best_row["C"],
            "gamma": best_row["gamma"],
            "kernel": "rbf",
            "class_weight": "balanced",
            "validation_accuracy": best_row["validation_accuracy"],
            "validation_macro_f1": best_row["validation_macro_f1"],
        },
        "existing_configuration": {
            "C": baseline_c,
            "gamma": baseline_gamma,
            "validation_accuracy": baseline_row["validation_accuracy"],
            "validation_macro_f1": baseline_row["validation_macro_f1"],
        },
        "validation_difference_selected_minus_existing": {
            "accuracy": (
                best_row["validation_accuracy"]
                - baseline_row["validation_accuracy"]
            ),
            "macro_f1": (
                best_row["validation_macro_f1"]
                - baseline_row["validation_macro_f1"]
            ),
        },
        "train_actor_ids": sorted(train_frame["actor_id"].unique().tolist()),
        "validation_actor_ids": sorted(
            validation_frame["actor_id"].unique().tolist()
        ),
        "training_recordings_used": len(train_paths),
        "validation_recordings_used": len(validation_paths),
        "test_set_used": False,
        "skipped_train_recordings": len(skipped_train),
        "skipped_validation_recordings": len(skipped_validation),
        "grid_size": len(result_rows),
        "artifacts": [
            "validation_grid_results.csv",
            "skipped_records.csv",
            "selected_svm_model.pkl",
            "training_only_scaler.pkl",
            "label_mapping.json",
        ],
    }
    (EXPERIMENT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Experiment artifacts saved under {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
