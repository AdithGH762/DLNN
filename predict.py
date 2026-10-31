from __future__ import annotations

import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import EMOTION_CODES, MAX_FRAMES, MODEL_DIR, N_MFCC, PROJECT_ROOT, SAMPLE_RATE
from src.preprocessing import (
    N_MFCC_DELTA,
    SVM_FEATURE_DIM,
    audio_quality,
    extract_mfcc,
    extract_mfcc_with_deltas,
    extract_svm_features,
    load_audio,
)

ORIGINAL_CNN_LSTM = "Original CNN-LSTM"
IMPROVED_CNN_LSTM = "Improved CNN-LSTM"
SVM_BASELINE = "SVM Baseline"
PREDICTION_MODELS = (ORIGINAL_CNN_LSTM, IMPROVED_CNN_LSTM, SVM_BASELINE)
_MFCC_HOP_LENGTH = 160
CNN_MAX_AUDIO_SECONDS = MAX_FRAMES * _MFCC_HOP_LENGTH / SAMPLE_RATE

MODEL_ARTIFACTS = {
    ORIGINAL_CNN_LSTM: ("emotionsense_cnn_lstm.keras", "label_mapping.json"),
    IMPROVED_CNN_LSTM: ("improved_cnn_lstm.keras", "label_mapping.json"),
    SVM_BASELINE: ("svm_model.pkl", "svm_scaler.pkl", "svm_label_mapping.json"),
}
_EMOTION_LABELS = frozenset(EMOTION_CODES.values())


def missing_model_artifacts(model_name: str) -> list[str]:
    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(f"Unsupported prediction model: {model_name}")
    return [
        filename for filename in MODEL_ARTIFACTS[model_name]
        if not _artifact_path(model_name, filename).is_file()
    ]


def _artifact_path(model_name: str, filename: str) -> Path:
    if model_name == IMPROVED_CNN_LSTM:
        return (
            PROJECT_ROOT
            / "experiments"
            / "improved_cnn_lstm_run_01"
            / filename
        )
    return MODEL_DIR / filename


@lru_cache(maxsize=2)
def _load_cnn_model(model_path: str, modified_ns: int, file_size: int):
    del modified_ns, file_size
    return tf.keras.models.load_model(model_path)


@lru_cache(maxsize=2)
def _load_svm_artifacts(model_path: str, model_modified_ns: int, model_size: int,
                        scaler_path: str,
                        scaler_modified_ns: int, scaler_size: int):
    del model_modified_ns, model_size, scaler_modified_ns, scaler_size
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler


@lru_cache(maxsize=4)
def _load_mapping(mapping_path: str, modified_ns: int, file_size: int) -> dict[str, str]:
    del modified_ns, file_size
    with open(mapping_path, encoding="utf-8") as mapping_file:
        mapping = json.load(mapping_file)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Invalid or empty label mapping: {mapping_path}")
    return mapping


def _cache_file_args(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _validate_mapping(mapping: dict[str, str]) -> None:
    expected_keys = {str(class_id) for class_id in range(len(_EMOTION_LABELS))}
    if (
        set(mapping) != expected_keys
        or any(not isinstance(label, str) for label in mapping.values())
        or set(mapping.values()) != _EMOTION_LABELS
    ):
        raise ValueError(
            "Label mapping must contain all eight RAVDESS emotions with "
            "contiguous zero-based class IDs."
        )


def _validate_cnn_model(model, feature_count: int) -> None:
    expected_input = (None, MAX_FRAMES, feature_count, 1)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(
            f"CNN-LSTM input shape {model.input_shape} does not match "
            f"the feature pipeline {expected_input}."
        )
    if tuple(model.output_shape) != (None, len(_EMOTION_LABELS)):
        raise ValueError(
            f"CNN-LSTM output shape {model.output_shape} does not provide "
            f"{len(_EMOTION_LABELS)} emotion scores."
        )


def _validate_scores(scores: np.ndarray, class_ids: np.ndarray) -> None:
    if (
        scores.ndim != 1
        or len(scores) != len(class_ids)
        or len(scores) != len(_EMOTION_LABELS)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1.0, rtol=0, atol=1e-5)
    ):
        raise ValueError("Model returned invalid or incomplete emotion scores.")


def _load_model_for_prediction(model_name: str):
    if model_name in {ORIGINAL_CNN_LSTM, IMPROVED_CNN_LSTM}:
        model_path = _artifact_path(model_name, MODEL_ARTIFACTS[model_name][0])
        model = _load_cnn_model(*_cache_file_args(model_path))
        feature_count = N_MFCC if model_name == ORIGINAL_CNN_LSTM else N_MFCC_DELTA
        _validate_cnn_model(model, feature_count)
        return model
    if model_name == SVM_BASELINE:
        model_path = _artifact_path(model_name, MODEL_ARTIFACTS[model_name][0])
        scaler_path = _artifact_path(model_name, MODEL_ARTIFACTS[model_name][1])
        model, scaler = _load_svm_artifacts(
            *_cache_file_args(model_path), *_cache_file_args(scaler_path)
        )
        if getattr(scaler, "n_features_in_", None) != SVM_FEATURE_DIM:
            raise ValueError(
                f"SVM scaler expects {getattr(scaler, 'n_features_in_', 'unknown')} "
                f"features; preprocessing produces {SVM_FEATURE_DIM}."
            )
        if not np.array_equal(
            np.asarray(getattr(model, "classes_", [])),
            np.arange(len(_EMOTION_LABELS)),
        ):
            raise ValueError("SVM classes do not match the eight-class label mapping.")
        return model, scaler
    raise ValueError(f"Unsupported prediction model: {model_name}")


def prediction_model_status(model_name: str) -> tuple[bool, str]:
    """Validate all required artifacts and a small end-to-end inference probe."""
    missing = missing_model_artifacts(model_name)
    if missing:
        return False, f"Missing required artifact(s): {', '.join(missing)}"
    try:
        mapping_path = _artifact_path(model_name, MODEL_ARTIFACTS[model_name][-1])
        mapping = _load_mapping(*_cache_file_args(mapping_path))
        _validate_mapping(mapping)
        if model_name == SVM_BASELINE:
            model, scaler = _load_model_for_prediction(model_name)
            class_ids = np.asarray(model.classes_)
            sample = scaler.transform(
                np.zeros((1, SVM_FEATURE_DIM), dtype=np.float32)
            )
            scores = np.asarray(model.predict_proba(sample)[0], dtype=np.float64)
        else:
            model = _load_model_for_prediction(model_name)
            feature_count = N_MFCC if model_name == ORIGINAL_CNN_LSTM else N_MFCC_DELTA
            sample = np.zeros(
                (1, MAX_FRAMES, feature_count, 1), dtype=np.float32
            )
            scores = np.asarray(model.predict(sample, verbose=0)[0], dtype=np.float64)
            class_ids = np.arange(len(scores))
        _validate_scores(scores, class_ids)
        if {str(int(class_id)) for class_id in class_ids} != set(mapping):
            raise ValueError("Model output classes do not match the label mapping.")
    except Exception as exc:
        return False, f"Artifact validation failed: {exc}"
    return True, "Ready for uploaded-audio prediction."


def available_prediction_models() -> list[str]:
    return [
        model_name for model_name in PREDICTION_MODELS
        if prediction_model_status(model_name)[0]
    ]


def _rank_scores(scores: np.ndarray, class_ids: np.ndarray,
                 mapping: dict[str, str]) -> list[dict[str, float | str]]:
    _validate_scores(scores, class_ids)
    ranked = []
    for class_id, score in zip(class_ids, scores):
        mapping_key = str(int(class_id))
        if mapping_key not in mapping:
            raise ValueError(f"Label mapping is missing class index {mapping_key}.")
        ranked.append({
            "emotion": mapping[mapping_key],
            "score": float(score),
        })
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def predict_file(audio_path: str | Path,
                 model_name: str = ORIGINAL_CNN_LSTM) -> dict:
    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(f"Unsupported prediction model: {model_name}")
    missing = missing_model_artifacts(model_name)
    if missing:
        raise FileNotFoundError(
            f"{model_name} is unavailable. Missing artifact(s): {', '.join(missing)}"
        )

    audio = load_audio(audio_path)
    duration_seconds = float(len(audio) / SAMPLE_RATE)

    mapping_path = _artifact_path(model_name, MODEL_ARTIFACTS[model_name][-1])
    mapping = _load_mapping(*_cache_file_args(mapping_path))
    _validate_mapping(mapping)

    if model_name in {ORIGINAL_CNN_LSTM, IMPROVED_CNN_LSTM}:
        model = _load_model_for_prediction(model_name)
        feature_count = N_MFCC if model_name == ORIGINAL_CNN_LSTM else N_MFCC_DELTA
        feature_extractor = (
            extract_mfcc if model_name == ORIGINAL_CNN_LSTM
            else extract_mfcc_with_deltas
        )
        features = feature_extractor(audio)
        if features.shape != (MAX_FRAMES, feature_count, 1):
            raise ValueError(
                f"{model_name} feature extractor returned {features.shape}; expected "
                f"({MAX_FRAMES}, {feature_count}, 1)."
            )
        scores = np.asarray(
            model.predict(np.expand_dims(features, axis=0), verbose=0)[0],
            dtype=np.float64,
        )
        class_ids = np.arange(len(scores))
        score_description = "Uncalibrated softmax score"
        processed_duration_seconds = min(
            duration_seconds, CNN_MAX_AUDIO_SECONDS
        )
    elif model_name == SVM_BASELINE:
        model, scaler = _load_model_for_prediction(model_name)
        features = extract_svm_features(audio)
        if features.shape != (SVM_FEATURE_DIM,):
            raise ValueError(
                f"SVM feature extractor returned {features.shape}; expected "
                f"({SVM_FEATURE_DIM},)."
            )
        scaled_features = scaler.transform(features.reshape(1, -1))
        scores = np.asarray(model.predict_proba(scaled_features)[0], dtype=np.float64)
        class_ids = np.asarray(model.classes_)
        score_description = "SVM probability estimate (not calibration-validated)"
        processed_duration_seconds = duration_seconds
    else:
        raise ValueError(f"Unsupported prediction model: {model_name}")

    ranked_scores = _rank_scores(scores, class_ids, mapping)
    if not ranked_scores:
        raise ValueError(f"{model_name} returned no class scores.")

    return {
        "model": model_name,
        "predicted_emotion": ranked_scores[0]["emotion"],
        "top_score": ranked_scores[0]["score"],
        "score_description": score_description,
        "class_scores": ranked_scores,
        "audio_quality": audio_quality(audio),
        "audio_duration_seconds": duration_seconds,
        "processed_duration_seconds": processed_duration_seconds,
        "audio_truncated": processed_duration_seconds < duration_seconds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file.")
    parser.add_argument("audio_path")
    parser.add_argument(
        "--model",
        choices=PREDICTION_MODELS,
        default=SVM_BASELINE,
        help="Prediction model to use (default: SVM Baseline).",
    )
    args = parser.parse_args()
    print(json.dumps(predict_file(args.audio_path, args.model), indent=2))
