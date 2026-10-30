from pathlib import Path

import numpy as np
import pytest

import predict
from config import MODEL_DIR


@pytest.fixture(autouse=True)
def clear_prediction_caches():
    predict._load_cnn_model.cache_clear()
    predict._load_svm_artifacts.cache_clear()
    predict._load_mapping.cache_clear()
    yield
    predict._load_cnn_model.cache_clear()
    predict._load_svm_artifacts.cache_clear()
    predict._load_mapping.cache_clear()


def _create_artifacts(directory: Path, filenames: tuple[str, ...]) -> None:
    for filename in filenames:
        (directory / filename).write_bytes(b"test artifact")


def test_available_prediction_models_requires_complete_artifact_sets(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    _create_artifacts(tmp_path, predict.MODEL_ARTIFACTS[predict.SVM_BASELINE])

    assert predict.available_prediction_models() == [predict.SVM_BASELINE]
    assert predict.missing_model_artifacts(predict.ORIGINAL_CNN_LSTM) == [
        "emotionsense_cnn_lstm.keras",
        "label_mapping.json",
    ]


def test_predict_file_reports_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="svm_scaler.pkl"):
        predict.predict_file("unused.wav", predict.SVM_BASELINE)


def test_cnn_model_loader_caches_model(monkeypatch, tmp_path):
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"model")
    calls = []
    model_instance = object()

    def fake_load_model(path):
        calls.append(path)
        return model_instance

    monkeypatch.setattr(predict.tf.keras.models, "load_model", fake_load_model)
    cache_args = predict._cache_file_args(model_path)

    assert predict._load_cnn_model(*cache_args) is model_instance
    assert predict._load_cnn_model(*cache_args) is model_instance
    assert calls == [str(model_path)]


def test_svm_prediction_uses_training_feature_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    _create_artifacts(tmp_path, predict.MODEL_ARTIFACTS[predict.SVM_BASELINE])
    audio = np.ones(64000, dtype=np.float32)
    features = np.arange(160, dtype=np.float32)
    transform_inputs = []

    class FakeScaler:
        def transform(self, values):
            transform_inputs.append(values.copy())
            return values

    class FakeSvm:
        classes_ = np.array([0, 2])

        def predict_proba(self, values):
            assert values.shape == (1, 160)
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(
        predict, "_load_svm_artifacts",
        lambda *args: (FakeSvm(), FakeScaler()),
    )
    monkeypatch.setattr(
        predict, "_load_mapping",
        lambda *args: {"0": "angry", "2": "disgust"},
    )
    monkeypatch.setattr(predict, "load_audio", lambda _: audio)
    monkeypatch.setattr(
        predict, "extract_svm_features",
        lambda value: features if value is audio else pytest.fail("unexpected audio"),
    )
    monkeypatch.setattr(predict, "audio_quality", lambda _: {"duration_seconds": 4.0})

    result = predict.predict_file("uploaded.wav", predict.SVM_BASELINE)

    assert result["model"] == predict.SVM_BASELINE
    assert result["predicted_emotion"] == "disgust"
    assert result["top_score"] == pytest.approx(0.8)
    assert result["score_description"] == (
        "SVM probability estimate (not calibration-validated)"
    )
    assert result["processed_duration_seconds"] == pytest.approx(4.0)
    assert result["audio_truncated"] is False
    np.testing.assert_array_equal(transform_inputs[0], features.reshape(1, -1))


def test_cnn_prediction_reports_truncation_and_uncalibrated_score(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    _create_artifacts(tmp_path, predict.MODEL_ARTIFACTS[predict.ORIGINAL_CNN_LSTM])
    audio = np.ones(64000, dtype=np.float32)

    class FakeModel:
        def predict(self, features, verbose=0):
            assert features.shape == (1, 200, 40, 1)
            return np.array([[0.75, 0.25]])

    monkeypatch.setattr(predict, "_load_cnn_model", lambda *args: FakeModel())
    monkeypatch.setattr(
        predict, "_load_mapping",
        lambda *args: {"0": "happy", "1": "sad"},
    )
    monkeypatch.setattr(predict, "load_audio", lambda _: audio)
    monkeypatch.setattr(
        predict, "extract_mfcc",
        lambda value: np.zeros((200, 40, 1), dtype=np.float32),
    )
    monkeypatch.setattr(predict, "audio_quality", lambda _: {})

    result = predict.predict_file("uploaded.wav", predict.ORIGINAL_CNN_LSTM)

    assert result["predicted_emotion"] == "happy"
    assert result["score_description"] == "Uncalibrated softmax score"
    assert result["audio_duration_seconds"] == pytest.approx(4.0)
    assert result["processed_duration_seconds"] == pytest.approx(
        predict.CNN_MAX_AUDIO_SECONDS
    )
    assert result["audio_truncated"] is True


@pytest.mark.skipif(
    any(not (MODEL_DIR / name).is_file()
        for name in predict.MODEL_ARTIFACTS[predict.SVM_BASELINE]),
    reason="Saved SVM inference artifacts are unavailable",
)
def test_svm_prediction_with_saved_artifacts(monkeypatch):
    audio = np.random.default_rng(3).normal(0, 0.1, 16000).astype(np.float32)
    monkeypatch.setattr(predict, "load_audio", lambda _: audio)

    result = predict.predict_file("synthetic.wav", predict.SVM_BASELINE)

    assert result["model"] == predict.SVM_BASELINE
    assert result["predicted_emotion"] in predict._load_mapping(
        *predict._cache_file_args(MODEL_DIR / "svm_label_mapping.json")
    ).values()
    assert len(result["class_scores"]) == 8
    assert sum(item["score"] for item in result["class_scores"]) == pytest.approx(1.0)
    assert result["audio_truncated"] is False
