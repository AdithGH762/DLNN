from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless and Streamlit
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
from src.preprocessing import load_audio, extract_mfcc
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

COND_DISPLAY = {
    "clean":    "Clean",
    "snr_20db": "20 dB SNR",
    "snr_10db": "10 dB SNR",
    "snr_0db":  "0 dB SNR",
}


def _bar_chart(results: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Save a styled bar chart for a given metric across noise conditions."""
    cond_order = ["clean", "snr_20db", "snr_10db", "snr_0db"]
    ordered = results.set_index("condition").reindex(
        [c for c in cond_order if c in results["condition"].values]
    ).reset_index()
    labels = [COND_DISPLAY.get(c, c) for c in ordered["condition"]]
    vals   = ordered[metric].tolist()
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"][:len(vals)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{metric.replace('_', ' ').title()} by Noise Condition", fontsize=13, pad=10)
    ax.set_xlabel("Test audio condition", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.012,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _learning_curves(history_path: Path, out_path: Path) -> None:
    """Plot training and validation loss/accuracy from the saved CSV."""
    if not history_path.exists():
        return
    hist = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist) + 1)

    for ax, metric, label in [
        (axes[0], "loss",     "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        if metric in hist.columns:
            ax.plot(epochs, hist[metric],     label="Train", linewidth=1.8)
        val_col = f"val_{metric}"
        if val_col in hist.columns:
            ax.plot(epochs, hist[val_col], label="Validation", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Training {label}")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)

    fig.suptitle("EmotionSense — Learning Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    model_path   = MODEL_DIR / "emotionsense_cnn_lstm.keras"
    mapping_path = MODEL_DIR / "label_mapping.json"
    if not model_path.exists() or not mapping_path.exists():
        raise SystemExit("Trained model or label mapping missing. Run `python train.py` first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model   = tf.keras.models.load_model(model_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (
        isinstance(model.output_shape, list)
        or not model.output_shape
        or model.output_shape[-1] is None
    ):
        raise ValueError("Original CNN-LSTM must expose one output with a fixed class count.")
    model_class_ids = range(int(model.output_shape[-1]))

    manifest_path = OUTPUT_DIR / "test_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit("Test manifest missing. Run `python train.py` first.")
    manifest = pd.read_csv(manifest_path)
    if manifest.empty or not {"path", "y_true"}.issubset(manifest.columns):
        raise ValueError("test_manifest.csv must contain non-empty path and y_true columns.")
    y_true_all = np.asarray(
        validate_test_manifest_labels(
            manifest[["path", "y_true"]].itertuples(index=False, name=None),
            mapping,
            model_class_ids,
            manifest_path.name,
        ),
        dtype=np.int64,
    )
    labels = [mapping[str(i)] for i in range(len(mapping))]
    provenance = create_run_provenance(
        evaluator="noise_evaluation",
        project_root=PROJECT_ROOT,
        input_files={
            "model_weights": {
                "path": model_path, "role": "inference_model", "required": True,
                "consumed_by_evaluator": True,
            },
            "label_mapping": {
                "path": mapping_path, "role": "class_id_to_emotion_mapping",
                "required": True, "consumed_by_evaluator": True,
            },
            "test_manifest": {
                "path": manifest_path, "role": "evaluation_samples_and_labels",
                "required": True, "consumed_by_evaluator": True,
            },
            "actor_split": {
                "path": MODEL_DIR / "split_actors.json", "role": "split_context",
                "required": False, "consumed_by_evaluator": False,
            },
            "project_config": {
                "path": PROJECT_ROOT / "config.py", "role": "runtime_configuration",
                "required": True, "consumed_by_evaluator": True,
            },
            "dependency_specification": {
                "path": PROJECT_ROOT / "requirements.txt",
                "role": "declared_dependency_constraints", "required": False,
                "consumed_by_evaluator": False,
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
            "training_configuration": {
                "path": PROJECT_ROOT / "train.py", "role": "model_training_configuration",
                "required": False, "consumed_by_evaluator": False,
            },
        },
        settings={
            "models": ["original"],
            "conditions": {"clean": None, "snr_20db": 20, "snr_10db": 10, "snr_0db": 0},
            "base_seed": 1000,
            "noise_seed_derivation": "SHA-256(base_seed, resolved recording path, condition)",
            "noise_method": "additive white Gaussian noise scaled to requested SNR",
            "sample_rate_hz": SAMPLE_RATE,
            "mfcc_coefficients": N_MFCC,
            "max_frames": MAX_FRAMES,
            "batch_size": 32,
            "cohort_policy": "require identical evaluated path and semantic-label sets across conditions",
        },
        package_names=(
            "numpy", "pandas", "scikit-learn", "librosa", "tensorflow",
            "matplotlib", "seaborn", "scipy", "soundfile", "joblib",
        ),
    )
    run_id = provenance["run_id"]
    provenance_path = run_provenance_path(OUTPUT_DIR, "noise_evaluation", run_id)
    write_run_provenance(provenance_path, provenance)
    print(f"Loaded model. Classes: {labels}")
    print(f"Test set: {len(manifest)} files.")

    records: list[dict] = []
    sample_records: list[dict] = []
    conditions = [("clean", None), ("snr_20db", 20), ("snr_10db", 10), ("snr_0db", 0)]

    for condition, snr in conditions:
        print(f"\n--- Condition: {condition} ---")
        features: list[np.ndarray] = []
        valid_idx: list[int] = []
        condition_samples = [
            {
                "model_key": "original",
                "run_id": run_id,
                "condition": condition,
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
                    audio, condition, snr, row.path, base_seed=1000
                )
                if noise_seed is not None:
                    condition_samples[i]["noise_base_seed"] = "1000"
                    condition_samples[i]["noise_seed"] = str(noise_seed)
                features.append(extract_mfcc(audio))
                valid_idx.append(i)
                condition_samples[i]["status"] = "features_ready"
            except Exception as exc:
                condition_samples[i]["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  Skipping {Path(row.path).name}: {exc}")

        if not features:
            sample_records.extend(condition_samples)
            print(f"  No usable files for {condition}.")
            continue

        predicted: dict[int, int] = {}
        inference_errors: dict[int, str] = {}
        try:
            probabilities = model.predict(
                np.stack(features), batch_size=32, verbose=0
            )
            if len(probabilities) != len(valid_idx):
                raise ValueError("Model returned a different number of rows than its input.")
            predicted.update(
                (manifest_idx, int(np.argmax(probability)))
                for manifest_idx, probability in zip(valid_idx, probabilities)
            )
        except Exception as batch_exc:
            for manifest_idx, feature in zip(valid_idx, features):
                try:
                    probability = model.predict(
                        np.expand_dims(feature, axis=0), batch_size=1, verbose=0
                    )
                    if len(probability) != 1:
                        raise ValueError("Model did not return exactly one prediction.")
                    predicted[manifest_idx] = int(np.argmax(probability[0]))
                except Exception as exc:
                    inference_errors[manifest_idx] = f"{type(exc).__name__}: {exc}"
            if inference_errors:
                print(
                    f"  Batch inference failed ({type(batch_exc).__name__}); "
                    f"{len(inference_errors)} recording(s) also failed individually."
                )
            else:
                print(
                    f"  Batch inference required per-record retries: "
                    f"{type(batch_exc).__name__}: {batch_exc}"
                )

        for manifest_idx, error in inference_errors.items():
            condition_samples[manifest_idx]["status"] = "inference_error"
            condition_samples[manifest_idx]["error"] = error
        evaluated_idx = [i for i in valid_idx if i in predicted]
        for manifest_idx in evaluated_idx:
            condition_samples[manifest_idx]["status"] = "evaluated"
            condition_samples[manifest_idx]["y_pred"] = predicted[manifest_idx]
            condition_samples[manifest_idx]["predicted_label"] = labels[
                predicted[manifest_idx]
            ]
        sample_records.extend(condition_samples)
        if not evaluated_idx:
            print(f"  No recordings completed inference for {condition}.")
            continue

        true_sub = y_true_all[evaluated_idx]
        pred = np.asarray([predicted[i] for i in evaluated_idx], dtype=np.int64)
        result     = classification_metrics(true_sub, pred, labels)

        print(f"  Accuracy={result['accuracy']:.4f}  "
              f"P={result['macro_precision']:.4f}  "
              f"R={result['macro_recall']:.4f}  "
              f"F1={result['macro_f1']:.4f}")

        records.append({
            "run_id": run_id,
            "condition":       condition,
            "n_test_files":    int(len(evaluated_idx)),
            "accuracy":        result["accuracy"],
            "macro_precision": result["macro_precision"],
            "macro_recall":    result["macro_recall"],
            "macro_f1":        result["macro_f1"],
        })

        records[-1]["_classification_report"] = result["classification_report"]
        records[-1]["_confusion_matrix"] = result["confusion_matrix"]

    write_run_provenance(provenance_path, provenance)
    prediction_path = OUTPUT_DIR / "noise_evaluation_predictions.csv"
    pd.DataFrame(sample_records).to_csv(prediction_path, index=False)
    if not evaluation_samples_match(
        sample_records,
        ["condition"],
        expected_groups=[(condition,) for condition, _ in conditions],
    ):
        raise SystemExit(
            "Evaluation identities or true labels differ across noise conditions, "
            f"or a recording failed in only some conditions. Per-record details were "
            f"saved to {prediction_path.name}; aggregate robustness outputs were not updated."
        )
    if not records:
        raise SystemExit("No evaluation results produced.")

    visual_artifacts = []
    for record in records:
        condition = record["condition"]
        visual_artifacts.append((
            condition,
            record.pop("_classification_report"),
            record.pop("_confusion_matrix"),
        ))

    results = pd.DataFrame(records)

    # Add delta columns relative to clean
    if "clean" in results["condition"].values:
        clean_row = results.loc[results["condition"] == "clean"].iloc[0]
        results["accuracy_change_vs_clean"] = results["accuracy"] - clean_row["accuracy"]
        results["macro_f1_change_vs_clean"] = results["macro_f1"] - clean_row["macro_f1"]

    results.to_csv(OUTPUT_DIR / "noise_evaluation_metrics.csv", index=False)
    results.to_json(OUTPUT_DIR / "noise_evaluation_metrics.json", orient="records", indent=2)

    for condition, classification_report, confusion_matrix in visual_artifacts:
        pd.DataFrame(classification_report).T.to_csv(
            OUTPUT_DIR / f"classification_report_{condition}.csv"
        )
        cm_arr = np.array(confusion_matrix)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm_arr, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted emotion", fontsize=11)
        ax.set_ylabel("True emotion", fontsize=11)
        ax.set_title(
            f"EmotionSense confusion matrix — {COND_DISPLAY.get(condition, condition)}",
            fontsize=12,
        )
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_matrix_{condition}.png", dpi=160)
        plt.close(fig)

    print("\n=== Summary ===")
    print(results[["condition", "n_test_files", "accuracy", "macro_f1"]].to_string(index=False))

    # Bar charts — use plain matplotlib, not seaborn barplot (API changed in 0.13)
    _bar_chart(results, "accuracy",  OUTPUT_DIR / "accuracy_by_noise.png")
    _bar_chart(results, "macro_f1",  OUTPUT_DIR / "macro_f1_by_noise.png")

    # Learning curves from training history
    _learning_curves(OUTPUT_DIR / "training_history.csv",
                     OUTPUT_DIR / "learning_curves.png")

    finish_run_provenance(provenance_path)
    print(f"\nAll evaluation outputs saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
