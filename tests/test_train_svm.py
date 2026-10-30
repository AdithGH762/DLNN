import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import train_svm


def test_build_xy_keeps_features_labels_and_paths_aligned_when_audio_is_skipped(
    monkeypatch,
):
    frame = pd.DataFrame(
        [
            {"path": "unreadable.wav", "emotion": "angry"},
            {"path": "readable.wav", "emotion": "calm"},
        ]
    )
    encoder = LabelEncoder().fit(["angry", "calm"])
    audio_by_path = {"readable.wav": np.array([2.0], dtype=np.float32)}

    def fake_load_audio(path):
        if path not in audio_by_path:
            raise ValueError("synthetic unreadable recording")
        return audio_by_path[path]

    monkeypatch.setattr(train_svm, "load_audio", fake_load_audio)
    monkeypatch.setattr(
        train_svm,
        "extract_svm_features",
        lambda audio: np.array([audio[0]], dtype=np.float32),
    )

    features, labels, paths = train_svm.build_xy(frame, encoder)

    np.testing.assert_array_equal(features, np.array([[2.0]], dtype=np.float32))
    np.testing.assert_array_equal(labels, np.array([encoder.transform(["calm"])[0]]))
    assert paths == ["readable.wav"]
