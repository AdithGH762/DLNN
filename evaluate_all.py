"""evaluate_all.py — Compare all EmotionSense models across noise conditions.

Evaluates any combination of:
  - Original CNN-LSTM (models/emotionsense_cnn_lstm.keras)
  - Improved CNN-LSTM with delta features + augmentation (models/improved_cnn_lstm.keras)
  - SVM baseline (models/svm_model.pkl)

For each model, runs evaluation under:
  clean / 20 dB SNR / 10 dB SNR / 0 dB SNR

Saves:
  outputs/all_models_comparison.csv / .json
  outputs/[model]_confusion_matrix_[condition].png
  outputs/[model]_classification_report_[condition].csv
  outputs/accuracy_comparison.png / macro_f1_comparison.png

Run:   python evaluate_all.py
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from config import (
    MAX_FRAMES,
    N_MFCC,
    OUTPUT_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    SAMPLE_RATE,
)
from src.dataset import validate_test_manifest_labels
from src.preprocessing import load_audio, extract_mfcc, extract_mfcc_with_deltas, extract_svm_features
from src.noise import add_noise_for_condition
from src.metrics import classification_metrics, evaluation_samples_match
from src.provenance import (
    create_run_provenance,
    finish_run_provenance,
    input_key,
    register_run_input,
    run_provenance_path,
    write_run_provenance,
)

SNR_CONDITIONS = [("clean", None), ("snr_20db", 20), ("snr_10db", 10), ("snr_0db", 0)]
COND_DISPLAY   = {"clean": "Clean", "snr_20db": "20 dB", "snr_10db": "10 dB", "snr_0db": "0 dB"}

MODEL_CONFIGS = [
    {
        "key":      "original",
        "label":    "Original CNN-LSTM",
        "type":     "keras",
        "model_path":   MODEL_DIR / "emotionsense_cnn_lstm.keras",
        "mapping_path": MODEL_DIR / "label_mapping.json",
        "manifest_path": OUTPUT_DIR / "test_manifest.csv",
        "feat_fn":  extract_mfcc,
    },
    {
        "key":      "improved",
        "label":    "Improved CNN-LSTM",
        "type":     "keras",
        "model_path":   MODEL_DIR / "improved_cnn_lstm.keras",
        "mapping_path": MODEL_DIR / "improved_label_mapping.json",
        "manifest_path": OUTPUT_DIR / "improved_test_manifest.csv",
        "feat_fn":  extract_mfcc_with_deltas,
    },
    {
        "key":      "svm",
        "label":    "SVM Baseline",
        "type":     "svm",
        "model_path":   MODEL_DIR / "svm_model.pkl",
        "scaler_path":  MODEL_DIR / "svm_scaler.pkl",
        "mapping_path": MODEL_DIR / "svm_label_mapping.json",
        "manifest_path": OUTPUT_DIR / "test_manifest.csv",
        "feat_fn":  extract_svm_features,
    },
]


def _confusion_fig(cm: np.ndarray, labels: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def _load_validated_manifest(
    cfg: dict, mapping: dict[str, str], model_class_ids
) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(cfg["manifest_path"])
    if manifest.empty or not {"path", "y_true"}.issubset(manifest.columns):
        raise ValueError(
            f"{cfg['manifest_path'].name} must contain non-empty path and y_true columns."
        )
    y_true = validate_test_manifest_labels(
        manifest[["path", "y_true"]].itertuples(index=False, name=None),
        mapping,
        model_class_ids,
        cfg["manifest_path"].name,
    )
    return manifest, np.asarray(y_true, dtype=np.int64)


def evaluate_keras(
    cfg: dict, sample_records: list[dict], run_id: str, provenance: dict
) -> list[dict]:
    model   = tf.keras.models.load_model(cfg["model_path"])
    mapping = json.loads(cfg["mapping_path"].read_text(encoding="utf-8"))
    if (
        isinstance(model.output_shape, list)
        or not model.output_shape
        or model.output_shape[-1] is None
    ):
        raise ValueError(f"{cfg['label']} must expose one output with a fixed class count.")
    model_class_ids = range(int(model.output_shape[-1]))
    manifest, y_true_all = _load_validated_manifest(cfg, mapping, model_class_ids)
    labels  = [mapping[str(i)] for i in range(len(mapping))]
    feat_fn = cfg["feat_fn"]
    key     = cfg["key"]
    records = []

    for cond, snr in SNR_CONDITIONS:
        feats, idx = [], []
        condition_samples = [
            {
                "model_key": key,
                "run_id": run_id,
                "condition": cond,
                "path": str(row.path),
                "y_true": int(y_true_all[i]),
                "true_label": labels[int(y_true_all[i])],
                "y_pred": "",
                "predicted_label": "",
                "noise_base_seed": "",
                "noise_seed": "",
                "recording_sha256": "",
                "status": "preprocessing_error",
                "error": "",
            }
            for i, row in enumerate(manifest.itertuples(index=False))
        ]
        for i, row in enumerate(manifest.itertuples(index=False)):
            try:
                recording_path = Path(row.path)
                fingerprint_key = input_key(recording_path, PROJECT_ROOT)
                if fingerprint_key not in provenance["inputs"]:
                    register_run_input(
                        provenance,
                        fingerprint_key,
                        recording_path,
                        PROJECT_ROOT,
                        role="dataset_audio_recording",
                        required=True,
                        consumed_by_evaluator=True,
                    )
                condition_samples[i]["recording_sha256"] = provenance["inputs"][
                    fingerprint_key
                ]["sha256"] or ""
                audio = load_audio(row.path)
                audio, noise_seed = add_noise_for_condition(
                    audio, cond, snr, row.path, base_seed=2000
                )
                if noise_seed is not None:
                    condition_samples[i]["noise_base_seed"] = "2000"
                    condition_samples[i]["noise_seed"] = str(noise_seed)
                feats.append(feat_fn(audio))
                idx.append(i)
                condition_samples[i]["status"] = "features_ready"
            except Exception as e:
                condition_samples[i]["error"] = f"{type(e).__name__}: {e}"
                print(f"  Skipping {Path(row.path).name}: {e}")
        predicted: dict[int, int] = {}
        inference_errors: dict[int, str] = {}
        if feats:
            try:
                probabilities = model.predict(
                    np.stack(feats), batch_size=32, verbose=0
                )
                if len(probabilities) != len(idx):
                    raise ValueError(
                        "Model returned a different number of rows than its input."
                    )
                predicted.update(
                    (manifest_idx, int(np.argmax(probability)))
                    for manifest_idx, probability in zip(idx, probabilities)
                )
            except Exception as batch_exc:
                for manifest_idx, feature in zip(idx, feats):
                    try:
                        probability = model.predict(
                            np.expand_dims(feature, axis=0), batch_size=1, verbose=0
                        )
                        if len(probability) != 1:
                            raise ValueError(
                                "Model did not return exactly one prediction."
                            )
                        predicted[manifest_idx] = int(np.argmax(probability[0]))
                    except Exception as exc:
                        inference_errors[manifest_idx] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                print(
                    f"  [{cfg['label']}] {cond} batch inference required "
                    f"per-record retries: {type(batch_exc).__name__}: {batch_exc}"
                )

        for manifest_idx, error in inference_errors.items():
            condition_samples[manifest_idx]["status"] = "inference_error"
            condition_samples[manifest_idx]["error"] = error
        evaluated_idx = [manifest_idx for manifest_idx in idx if manifest_idx in predicted]
        for manifest_idx in evaluated_idx:
            condition_samples[manifest_idx]["status"] = "evaluated"
            condition_samples[manifest_idx]["y_pred"] = predicted[manifest_idx]
            condition_samples[manifest_idx]["predicted_label"] = labels[
                predicted[manifest_idx]
            ]
        sample_records.extend(condition_samples)
        if not evaluated_idx:
            continue
        idx_arr = np.asarray(evaluated_idx)
        true_sub = y_true_all[idx_arr]
        pred = np.asarray([predicted[manifest_idx] for manifest_idx in evaluated_idx])
        result    = classification_metrics(true_sub, pred, labels)
        print(f"  [{cfg['label']}] {cond}: acc={result['accuracy']:.4f}  f1={result['macro_f1']:.4f}")
        records.append({
            **_base_record(cfg, cond, idx_arr, result),
            "_labels": labels,
            "_classification_report": result["classification_report"],
            "_confusion_matrix": result["confusion_matrix"],
        })
    return records


def evaluate_svm(
    cfg: dict, sample_records: list[dict], run_id: str, provenance: dict
) -> list[dict]:
    with open(cfg["model_path"], "rb") as f:
        svm = pickle.load(f)
    with open(cfg["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    mapping = json.loads(cfg["mapping_path"].read_text(encoding="utf-8"))
    manifest, y_true_all = _load_validated_manifest(cfg, mapping, svm.classes_)
    labels  = [mapping[str(i)] for i in range(len(mapping))]
    feat_fn = cfg["feat_fn"]
    records = []

    for cond, snr in SNR_CONDITIONS:
        feats, idx = [], []
        condition_samples = [
            {
                "model_key": cfg["key"],
                "run_id": run_id,
                "condition": cond,
                "path": str(row.path),
                "y_true": int(y_true_all[i]),
                "true_label": labels[int(y_true_all[i])],
                "y_pred": "",
                "predicted_label": "",
                "noise_base_seed": "",
                "noise_seed": "",
                "recording_sha256": "",
                "status": "preprocessing_error",
                "error": "",
            }
            for i, row in enumerate(manifest.itertuples(index=False))
        ]
        for i, row in enumerate(manifest.itertuples(index=False)):
            try:
                recording_path = Path(row.path)
                fingerprint_key = input_key(recording_path, PROJECT_ROOT)
                if fingerprint_key not in provenance["inputs"]:
                    register_run_input(
                        provenance,
                        fingerprint_key,
                        recording_path,
                        PROJECT_ROOT,
                        role="dataset_audio_recording",
                        required=True,
                        consumed_by_evaluator=True,
                    )
                condition_samples[i]["recording_sha256"] = provenance["inputs"][
                    fingerprint_key
                ]["sha256"] or ""
                audio = load_audio(row.path)
                audio, noise_seed = add_noise_for_condition(
                    audio, cond, snr, row.path, base_seed=2000
                )
                if noise_seed is not None:
                    condition_samples[i]["noise_base_seed"] = "2000"
                    condition_samples[i]["noise_seed"] = str(noise_seed)
                feats.append(feat_fn(audio))
                idx.append(i)
                condition_samples[i]["status"] = "features_ready"
            except Exception as e:
                condition_samples[i]["error"] = f"{type(e).__name__}: {e}"
                print(f"  Skipping {Path(row.path).name}: {e}")
        predicted: dict[int, int] = {}
        inference_errors: dict[int, str] = {}
        if feats:
            try:
                scaled = scaler.transform(np.stack(feats))
                predictions = svm.predict(scaled)
                if len(predictions) != len(idx):
                    raise ValueError(
                        "SVM returned a different number of rows than its input."
                    )
                predicted.update(
                    (manifest_idx, int(prediction))
                    for manifest_idx, prediction in zip(idx, predictions)
                )
            except Exception as batch_exc:
                for manifest_idx, feature in zip(idx, feats):
                    try:
                        scaled = scaler.transform(np.expand_dims(feature, axis=0))
                        prediction = svm.predict(scaled)
                        if len(prediction) != 1:
                            raise ValueError(
                                "SVM did not return exactly one prediction."
                            )
                        predicted[manifest_idx] = int(prediction[0])
                    except Exception as exc:
                        inference_errors[manifest_idx] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                print(
                    f"  [{cfg['label']}] {cond} batch inference required "
                    f"per-record retries: {type(batch_exc).__name__}: {batch_exc}"
                )

        for manifest_idx, error in inference_errors.items():
            condition_samples[manifest_idx]["status"] = "inference_error"
            condition_samples[manifest_idx]["error"] = error
        evaluated_idx = [manifest_idx for manifest_idx in idx if manifest_idx in predicted]
        for manifest_idx in evaluated_idx:
            condition_samples[manifest_idx]["status"] = "evaluated"
            condition_samples[manifest_idx]["y_pred"] = predicted[manifest_idx]
            condition_samples[manifest_idx]["predicted_label"] = labels[
                predicted[manifest_idx]
            ]
        sample_records.extend(condition_samples)
        if not evaluated_idx:
            continue
        idx_arr = np.asarray(evaluated_idx)
        true_sub = y_true_all[idx_arr]
        pred = np.asarray([predicted[manifest_idx] for manifest_idx in evaluated_idx])
        result   = classification_metrics(true_sub, pred, labels)
        print(f"  [{cfg['label']}] {cond}: acc={result['accuracy']:.4f}  f1={result['macro_f1']:.4f}")
        records.append({
            **_base_record(cfg, cond, idx_arr, result),
            "_labels": labels,
            "_classification_report": result["classification_report"],
            "_confusion_matrix": result["confusion_matrix"],
        })
    return records


def _base_record(cfg: dict, cond: str, idx: np.ndarray, result: dict) -> dict:
    return {
        "model":           cfg["label"],
        "model_key":       cfg["key"],
        "condition":       cond,
        "n_test_files":    int(len(idx)),
        "accuracy":        result["accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall":    result["macro_recall"],
        "macro_f1":        result["macro_f1"],
    }


def _save_outputs(result: dict, labels: list[str], key: str, cond: str) -> None:
    cm  = np.array(result["confusion_matrix"])
    fig = _confusion_fig(cm, labels,
                         f"EmotionSense ({key}) — {COND_DISPLAY.get(cond, cond)}")
    fig.savefig(OUTPUT_DIR / f"{key}_confusion_matrix_{cond}.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(result["classification_report"]).T.to_csv(
        OUTPUT_DIR / f"{key}_classification_report_{cond}.csv"
    )


def _comparison_chart(results: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Grouped bar chart: x=noise condition, bars grouped by model."""
    cond_order = ["clean", "snr_20db", "snr_10db", "snr_0db"]
    models     = results["model_key"].unique().tolist()
    colors     = ["#4c72b0", "#55a868", "#c44e52"][:len(models)]
    x          = np.arange(len(cond_order))
    width      = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (mk, color) in enumerate(zip(models, colors)):
        sub = results[results["model_key"] == mk].set_index("condition")[metric]
        vals = [float(sub.get(c, np.nan)) for c in cond_order]
        bars = ax.bar(x + (i - len(models)/2 + 0.5) * width, vals,
                      width=width * 0.9, label=results.loc[results["model_key"]==mk, "model"].iloc[0],
                      color=color, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([COND_DISPLAY.get(c, c) for c in cond_order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"{metric.replace('_',' ').title()} — Model Comparison by Noise Condition",
                 fontsize=12, pad=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _provenance_inputs(selected_configs: list[dict]) -> dict[str, dict]:
    inputs = {
        "project_config": {
            "path": PROJECT_ROOT / "config.py", "role": "runtime_configuration",
            "required": True, "consumed_by_evaluator": True,
        },
        "dependency_specification": {
            "path": PROJECT_ROOT / "requirements.txt",
            "role": "declared_dependency_constraints", "required": False,
            "consumed_by_evaluator": False,
        },
        "actor_split": {
            "path": MODEL_DIR / "split_actors.json", "role": "split_context",
            "required": False, "consumed_by_evaluator": False,
        },
        "evaluation_code": {
            "path": Path(__file__), "role": "evaluation_implementation",
            "required": True, "consumed_by_evaluator": True,
        },
        "preprocessing_code": {
            "path": PROJECT_ROOT / "src" / "preprocessing.py",
            "role": "audio_and_feature_pipeline", "required": True,
            "consumed_by_evaluator": True,
        },
        "dataset_code": {
            "path": PROJECT_ROOT / "src" / "dataset.py",
            "role": "manifest_label_validation", "required": True,
            "consumed_by_evaluator": True,
        },
        "noise_code": {
            "path": PROJECT_ROOT / "src" / "noise.py",
            "role": "noise_generation", "required": True,
            "consumed_by_evaluator": True,
        },
        "metrics_code": {
            "path": PROJECT_ROOT / "src" / "metrics.py",
            "role": "metric_and_cohort_implementation", "required": True,
            "consumed_by_evaluator": True,
        },
        "provenance_code": {
            "path": PROJECT_ROOT / "src" / "provenance.py",
            "role": "run_metadata_and_input_fingerprinting", "required": True,
            "consumed_by_evaluator": True,
        },
    }
    training_scripts = {
        "original": "train.py",
        "improved": "train_improved.py",
        "svm": "train_svm.py",
    }
    for cfg in selected_configs:
        key = cfg["key"]
        inputs[f"{key}_model_weights"] = {
            "path": cfg["model_path"], "role": "inference_model",
            "required": True, "consumed_by_evaluator": True,
        }
        inputs[f"{key}_label_mapping"] = {
            "path": cfg["mapping_path"], "role": "class_id_to_emotion_mapping",
            "required": True, "consumed_by_evaluator": True,
        }
        inputs[f"{key}_test_manifest"] = {
            "path": cfg["manifest_path"], "role": "evaluation_samples_and_labels",
            "required": True, "consumed_by_evaluator": True,
        }
        inputs[f"{key}_training_configuration"] = {
            "path": PROJECT_ROOT / training_scripts[key],
            "role": "model_training_configuration", "required": False,
            "consumed_by_evaluator": False,
        }
        if cfg["type"] == "svm":
            inputs["svm_scaler"] = {
                "path": cfg["scaler_path"], "role": "inference_feature_scaler",
                "required": True, "consumed_by_evaluator": True,
            }
            inputs["svm_training_metadata"] = {
                "path": MODEL_DIR / "svm_config.json",
                "role": "saved_training_metadata", "required": False,
                "consumed_by_evaluator": False,
            }
    return inputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    sample_records: list[dict] = []
    evaluated_models: list[str] = []
    selected_configs = [
        cfg for cfg in MODEL_CONFIGS if cfg["model_path"].is_file()
    ]
    provenance = create_run_provenance(
        evaluator="all_models_evaluation",
        project_root=PROJECT_ROOT,
        input_files=_provenance_inputs(selected_configs),
        settings={
            "models": [
                {"key": cfg["key"], "type": cfg["type"], "features": cfg["feat_fn"].__name__}
                for cfg in selected_configs
            ],
            "skipped_models": [
                {
                    "key": cfg["key"],
                    "model_path": cfg["model_path"].relative_to(PROJECT_ROOT).as_posix(),
                    "status": "model_artifact_missing",
                }
                for cfg in MODEL_CONFIGS if not cfg["model_path"].is_file()
            ],
            "conditions": {"clean": None, "snr_20db": 20, "snr_10db": 10, "snr_0db": 0},
            "base_seed": 2000,
            "noise_seed_derivation": "SHA-256(base_seed, resolved recording path, condition)",
            "noise_method": "additive white Gaussian noise scaled to requested SNR",
            "sample_rate_hz": SAMPLE_RATE,
            "mfcc_coefficients": N_MFCC,
            "max_frames": MAX_FRAMES,
            "batch_size": 32,
            "cohort_policy": "require identical evaluated path and semantic-label sets across models and conditions",
        },
        package_names=(
            "numpy", "pandas", "scikit-learn", "librosa", "tensorflow",
            "matplotlib", "seaborn", "scipy", "soundfile", "joblib",
        ),
    )
    run_id = provenance["run_id"]
    provenance_path = run_provenance_path(
        OUTPUT_DIR, "all_models_evaluation", run_id
    )
    write_run_provenance(provenance_path, provenance)
    if not selected_configs:
        raise SystemExit("No models available for evaluation.")
    if provenance["missing_required_inputs"]:
        raise SystemExit(
            "Required evaluation inputs are missing: "
            + ", ".join(provenance["missing_required_inputs"])
            + f". Incomplete run metadata saved to {provenance_path.name}."
        )

    for cfg in MODEL_CONFIGS:
        if cfg not in selected_configs:
            print(f"[SKIP] {cfg['label']}: model not found at {cfg['model_path']}")
            continue
        print(f"\n{'='*60}")
        print(f" Evaluating: {cfg['label']}")
        print(f"{'='*60}")
        if cfg["type"] == "keras":
            model_records = evaluate_keras(cfg, sample_records, run_id, provenance)
        elif cfg["type"] == "svm":
            model_records = evaluate_svm(cfg, sample_records, run_id, provenance)
        else:
            continue
        if sample_records and any(
            row["model_key"] == cfg["key"] for row in sample_records
        ):
            evaluated_models.append(cfg["key"])
        all_records.extend(model_records)

    write_run_provenance(provenance_path, provenance)
    prediction_path = OUTPUT_DIR / "all_models_predictions.csv"
    pd.DataFrame(sample_records).to_csv(prediction_path, index=False)
    expected_groups = [
        (model_key, condition)
        for model_key in evaluated_models
        for condition, _ in SNR_CONDITIONS
    ]
    if not evaluation_samples_match(
        sample_records,
        ["model_key", "condition"],
        expected_groups=expected_groups,
    ):
        raise SystemExit(
            "Evaluated recording identities or true labels differ across models or "
            "noise conditions, or a recording failed in only some groups. Per-record "
            f"details were saved to {prediction_path.name}; comparison outputs were not updated."
        )

    if not all_records:
        raise SystemExit("No models available for evaluation.")

    for record in all_records:
        record["run_id"] = run_id

    visual_artifacts = []
    for record in all_records:
        visual_artifacts.append((
            record["model_key"],
            record["condition"],
            record.pop("_labels"),
            record.pop("_classification_report"),
            record.pop("_confusion_matrix"),
        ))

    results = pd.DataFrame(all_records)

    # Add delta columns vs clean within each model
    rows = []
    for mk in results["model_key"].unique():
        sub = results[results["model_key"] == mk].copy()
        clean_row = sub[sub["condition"] == "clean"]
        if not clean_row.empty:
            clean_acc = clean_row["accuracy"].iloc[0]
            clean_f1  = clean_row["macro_f1"].iloc[0]
            sub["accuracy_delta_vs_clean"] = sub["accuracy"] - clean_acc
            sub["macro_f1_delta_vs_clean"] = sub["macro_f1"] - clean_f1
        rows.append(sub)
    results = pd.concat(rows, ignore_index=True)

    results.to_csv(OUTPUT_DIR / "all_models_comparison.csv", index=False)
    results.to_json(OUTPUT_DIR / "all_models_comparison.json", orient="records", indent=2)

    for model_key, condition, labels, report, matrix in visual_artifacts:
        _save_outputs(
            {"classification_report": report, "confusion_matrix": matrix},
            labels,
            model_key,
            condition,
        )

    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(results[["model", "condition", "accuracy", "macro_f1"]].to_string(index=False))

    _comparison_chart(results, "accuracy",  OUTPUT_DIR / "accuracy_comparison.png")
    _comparison_chart(results, "macro_f1",  OUTPUT_DIR / "macro_f1_comparison.png")

    finish_run_provenance(provenance_path)
    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
