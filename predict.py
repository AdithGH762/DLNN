from __future__ import annotations

import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import MAX_FRAMES, MODEL_DIR, SAMPLE_RATE
from src.preprocessing import (
    audio_quality,
    extract_mfcc,
    extract_svm_features,
    load_audio,
)

ORIGINAL_CNN_LSTM = "Original CNN-LSTM"
SVM_BASELINE = "SVM Baseline"
PREDICTION_MODELS = (ORIGINAL_CNN_LSTM, SVM_BASELINE)
_MFCC_HOP_LENGTH = 160
CNN_MAX_AUDIO_SECONDS = MAX_FRAMES * _MFCC_HOP_LENGTH / SAMPLE_RATE

MODEL_ARTIFACTS = {
    ORIGINAL_CNN_LSTM: ("emotionsense_cnn_lstm.keras", "label_mapping.json"),
    SVM_BASELINE: ("svm_model.pkl", "svm_scaler.pkl", "svm_label_mapping.json"),
}


def missing_model_artifacts(model_name: str) -> list[str]:
    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(f"Unsupported prediction model: {model_name}")
    return [
        filename for filename in MODEL_ARTIFACTS[model_name]
        if not (MODEL_DIR / filename).is_file()
    ]


def available_prediction_models() -> list[str]:
    return [
        model_name for model_name in PREDICTION_MODELS
        if not missing_model_artifacts(model_name)
    ]


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


def _rank_scores(scores: np.ndarray, class_ids: np.ndarray,
                 mapping: dict[str, str]) -> list[dict[str, float | str]]:
    if scores.ndim != 1 or len(scores) != len(class_ids):
        raise ValueError("Model score count does not match its class labels.")
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
    missing = missing_model_artifacts(model_name)
    if missing:
        raise FileNotFoundError(
            f"{model_name} is unavailable. Missing artifact(s): {', '.join(missing)}"
        )

    audio = load_audio(audio_path)
    duration_seconds = float(len(audio) / SAMPLE_RATE)

    if model_name == ORIGINAL_CNN_LSTM:
        model_path, model_modified_ns, model_size = _cache_file_args(
            MODEL_DIR / MODEL_ARTIFACTS[model_name][0]
        )
        mapping_args = _cache_file_args(MODEL_DIR / MODEL_ARTIFACTS[model_name][1])
        model = _load_cnn_model(model_path, model_modified_ns, model_size)
        mapping = _load_mapping(*mapping_args)
        features = extract_mfcc(audio)
        scores = np.asarray(
            model.predict(np.expand_dims(features, axis=0), verbose=0)[0],
            dtype=np.float64,
        )
        class_ids = np.arange(len(scores))
        score_description = "Uncalibrated softmax score"
        processed_duration_seconds = min(
            duration_seconds, CNN_MAX_AUDIO_SECONDS
        )
    else:
        model_args = _cache_file_args(MODEL_DIR / MODEL_ARTIFACTS[model_name][0])
        scaler_args = _cache_file_args(MODEL_DIR / MODEL_ARTIFACTS[model_name][1])
        mapping_args = _cache_file_args(MODEL_DIR / MODEL_ARTIFACTS[model_name][2])
        model, scaler = _load_svm_artifacts(*model_args, *scaler_args)
        mapping = _load_mapping(*mapping_args)
        features = extract_svm_features(audio).reshape(1, -1)
        scaled_features = scaler.transform(features)
        scores = np.asarray(model.predict_proba(scaled_features)[0], dtype=np.float64)
        class_ids = np.asarray(model.classes_)
        score_description = "SVM probability estimate (not calibration-validated)"
        processed_duration_seconds = duration_seconds

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
        default=ORIGINAL_CNN_LSTM,
        help="Prediction model to use (default: Original CNN-LSTM).",
    )
    args = parser.parse_args()
    print(json.dumps(predict_file(args.audio_path, args.model), indent=2))
