from pathlib import Path

import numpy as np
import pytest

import predict
from config import DATA_DIR, MAX_FRAMES, MODEL_DIR, N_MFCC
from src.preprocessing import N_MFCC_DELTA, SVM_FEATURE_DIM

MAPPING = {
    "0": "angry",
    "1": "calm",
    "2": "disgust",
    "3": "fearful",
    "4": "happy",
    "5": "neutral",
    "6": "sad",
    "7": "surprised",
}


@pytest.fixture(autouse=True)
def clear_prediction_caches():
    predict._load_cnn_model.cache_clear()
    predict._load_svm_artifacts.cache_clear()
    predict._load_mapping.cache_clear()
    yield
    predict._load_cnn_model.cache_clear()
    predict._load_svm_artifacts.cache_clear()
    predict._load_mapping.cache_clear()


def _create_artifacts(directory: Path, model_name: str) -> None:
    for filename in predict.MODEL_ARTIFACTS[model_name]:
        path = predict._artifact_path(model_name, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test artifact")


class FakeCnn:
    output_shape = (None, 8)

    def __init__(self, feature_count):
        self.input_shape = (None, MAX_FRAMES, feature_count, 1)

    def predict(self, features, verbose=0):
        assert features.shape == (1, *self.input_shape[1:])
        scores = np.arange(1, 9, dtype=np.float64)
        return (scores / scores.sum()).reshape(1, -1)


class FakeScaler:
    n_features_in_ = SVM_FEATURE_DIM

    def transform(self, values):
        assert values.shape == (1, SVM_FEATURE_DIM)
        return values


class FakeSvm:
    classes_ = np.arange(8)

    def predict_proba(self, values):
        assert values.shape == (1, SVM_FEATURE_DIM)
        scores = np.arange(1, 9, dtype=np.float64)
        return (scores / scores.sum()).reshape(1, -1)


def _install_fake_artifact_loaders(monkeypatch):
    def load_cnn(path, *_):
        if "improved_cnn_lstm_run_01" in path:
            return FakeCnn(N_MFCC_DELTA)
        return FakeCnn(N_MFCC)

    monkeypatch.setattr(predict, "_load_cnn_model", load_cnn)
    monkeypatch.setattr(
        predict, "_load_svm_artifacts", lambda *_: (FakeSvm(), FakeScaler())
    )
    monkeypatch.setattr(predict, "_load_mapping", lambda *_: dict(MAPPING))
    monkeypatch.setattr(predict, "load_audio", lambda _: np.ones(16000, dtype=np.float32))
    monkeypatch.setattr(
        predict,
        "audio_quality",
        lambda _: {"duration_seconds": 1.0},
    )


def test_artifact_availability_requires_complete_artifact_sets(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict, "PROJECT_ROOT", tmp_path)
    _create_artifacts(tmp_path, predict.SVM_BASELINE)
    _create_artifacts(tmp_path, predict.IMPROVED_CNN_LSTM)

    assert predict.available_prediction_models() == []
    assert predict.missing_model_artifacts(predict.ORIGINAL_CNN_LSTM) == [
        "emotionsense_cnn_lstm.keras",
        "label_mapping.json",
    ]
    assert predict.missing_model_artifacts(predict.IMPROVED_CNN_LSTM) == []


def test_available_models_require_loadable_compatible_prediction_pipelines(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict, "PROJECT_ROOT", tmp_path)
    for model_name in predict.PREDICTION_MODELS:
        _create_artifacts(tmp_path, model_name)
    _install_fake_artifact_loaders(monkeypatch)

    assert predict.available_prediction_models() == list(predict.PREDICTION_MODELS)


def test_invalid_cnn_shape_is_not_reported_as_available(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict, "PROJECT_ROOT", tmp_path)
    _create_artifacts(tmp_path, predict.IMPROVED_CNN_LSTM)
    monkeypatch.setattr(
        predict, "_load_mapping", lambda *_: dict(MAPPING)
    )
    monkeypatch.setattr(
        predict, "_load_cnn_model",
        lambda *_: FakeCnn(N_MFCC),
    )

    available, message = predict.prediction_model_status(predict.IMPROVED_CNN_LSTM)

    assert not available
    assert "input shape" in message


def test_predict_file_reports_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="svm_scaler.pkl"):
        predict.predict_file("unused.wav", predict.SVM_BASELINE)


@pytest.mark.parametrize(
    ("model_name", "feature_count", "extractor_name", "expected_description"),
    [
        (
            predict.ORIGINAL_CNN_LSTM,
            N_MFCC,
            "extract_mfcc",
            "Uncalibrated softmax score",
        ),
        (
            predict.IMPROVED_CNN_LSTM,
            N_MFCC_DELTA,
            "extract_mfcc_with_deltas",
            "Uncalibrated softmax score",
        ),
        (
            predict.SVM_BASELINE,
            SVM_FEATURE_DIM,
            "extract_svm_features",
            "SVM probability estimate (not calibration-validated)",
        ),
    ],
)
def test_each_model_uses_its_own_features_and_returns_class_scores(
    monkeypatch, tmp_path, model_name, feature_count, extractor_name,
    expected_description,
):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict, "PROJECT_ROOT", tmp_path)
    _create_artifacts(tmp_path, model_name)
    _install_fake_artifact_loaders(monkeypatch)
    audio = np.ones(16000, dtype=np.float32)
    extracted = []

    def extract(features):
        assert features is audio
        extracted.append(features)
        if model_name == predict.SVM_BASELINE:
            return np.zeros(SVM_FEATURE_DIM, dtype=np.float32)
        return np.zeros((MAX_FRAMES, feature_count, 1), dtype=np.float32)

    monkeypatch.setattr(predict, "load_audio", lambda _: audio)
    monkeypatch.setattr(predict, extractor_name, extract)

    result = predict.predict_file("uploaded.wav", model_name)

    assert result["model"] == model_name
    assert result["predicted_emotion"] == "surprised"
    assert result["score_description"] == expected_description
    assert len(result["class_scores"]) == 8
    assert sum(item["score"] for item in result["class_scores"]) == pytest.approx(1.0)
    assert extracted == [audio]


@pytest.mark.parametrize("model_name", predict.PREDICTION_MODELS)
@pytest.mark.skipif(
    not (DATA_DIR / "Actor_01").is_dir(),
    reason="RAVDESS audio is unavailable",
)
def test_saved_model_predicts_real_recording(model_name):
    available, reason = predict.prediction_model_status(model_name)
    if not available:
        pytest.skip(reason)
    audio_path = next((DATA_DIR / "Actor_01").glob("*.wav"))

    result = predict.predict_file(audio_path, model_name)

    assert result["model"] == model_name
    assert result["predicted_emotion"] in set(MAPPING.values())
    assert len(result["class_scores"]) == 8
    assert all(np.isfinite(item["score"]) for item in result["class_scores"])
    assert sum(item["score"] for item in result["class_scores"]) == pytest.approx(1.0)
