import numpy as np
from src.preprocessing import extract_mfcc, audio_quality


def test_mfcc_shape_and_finite_values():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    features = extract_mfcc(audio)
    assert features.shape == (200, 40, 1)
    assert np.isfinite(features).all()


def test_silence_is_handled():
    features = extract_mfcc(np.zeros(4000, dtype=np.float32))
    assert features.shape == (200, 40, 1)
    assert np.isfinite(features).all()
    assert audio_quality(np.zeros(4000))["silence_like"] is True
