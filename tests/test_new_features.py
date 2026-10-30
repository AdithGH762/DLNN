"""Tests for new feature extraction functions and SVM baseline."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing import (
    extract_mfcc_with_deltas,
    extract_svm_features,
    N_MFCC_DELTA, SVM_FEATURE_DIM,
)
from config import N_MFCC, MAX_FRAMES, MODEL_DIR


# ── Shared synthetic audio ─────────────────────────────────────────────────────
SR = 16_000
_t = np.arange(SR, dtype=np.float32) / SR
_AUDIO = 0.25 * np.sin(2 * np.pi * 220 * _t)


# ── extract_mfcc_with_deltas ───────────────────────────────────────────────────

def test_delta_mfcc_shape():
    feat = extract_mfcc_with_deltas(_AUDIO)
    assert feat.shape == (MAX_FRAMES, N_MFCC * 3, 1), (
        f"Expected ({MAX_FRAMES}, {N_MFCC*3}, 1) got {feat.shape}"
    )


def test_delta_mfcc_finite():
    feat = extract_mfcc_with_deltas(_AUDIO)
    assert np.isfinite(feat).all(), "Delta MFCC features contain NaN or Inf"


def test_delta_mfcc_silence():
    silent = np.zeros(4000, dtype=np.float32)
    feat = extract_mfcc_with_deltas(silent)
    assert feat.shape == (MAX_FRAMES, N_MFCC * 3, 1)
    assert np.isfinite(feat).all()


def test_delta_mfcc_short_audio():
    """Very short clip should be padded, not crash."""
    short = _AUDIO[:SR // 4]   # 0.25 seconds
    feat  = extract_mfcc_with_deltas(short)
    assert feat.shape == (MAX_FRAMES, N_MFCC * 3, 1)
    assert np.isfinite(feat).all()


def test_delta_mfcc_long_audio():
    """Long clip should be truncated to MAX_FRAMES."""
    long = np.tile(_AUDIO, 5)  # 5 seconds
    feat = extract_mfcc_with_deltas(long)
    assert feat.shape == (MAX_FRAMES, N_MFCC * 3, 1)


# ── extract_svm_features ───────────────────────────────────────────────────────

def test_svm_feature_shape():
    feat = extract_svm_features(_AUDIO)
    assert feat.shape == (SVM_FEATURE_DIM,), (
        f"Expected ({SVM_FEATURE_DIM},) got {feat.shape}"
    )
    assert feat.shape == (N_MFCC * 4,)


def test_svm_feature_finite():
    feat = extract_svm_features(_AUDIO)
    assert np.isfinite(feat).all()


def test_svm_feature_silence():
    feat = extract_svm_features(np.zeros(4000, dtype=np.float32))
    assert feat.shape == (SVM_FEATURE_DIM,)
    assert np.isfinite(feat).all()


def test_svm_feature_dtype():
    feat = extract_svm_features(_AUDIO)
    assert feat.dtype == np.float32


# ── SVM model artifacts (only when model exists) ──────────────────────────────

@pytest.mark.skipif(
    not (MODEL_DIR / "svm_model.pkl").exists(),
    reason="SVM model not trained yet"
)
def test_svm_model_loadable():
    with open(MODEL_DIR / "svm_model.pkl", "rb") as f:
        svm = pickle.load(f)
    assert hasattr(svm, "predict"), "Loaded object is not a sklearn model"
    assert hasattr(svm, "predict_proba"), "SVM needs probability=True"


@pytest.mark.skipif(
    not (MODEL_DIR / "svm_scaler.pkl").exists(),
    reason="SVM scaler not trained yet"
)
def test_svm_scaler_loadable():
    with open(MODEL_DIR / "svm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, "transform")
    # Check scaler was fitted on N_MFCC*4 features
    assert scaler.n_features_in_ == N_MFCC * 4, (
        f"Scaler fitted on {scaler.n_features_in_} features, expected {N_MFCC*4}"
    )


@pytest.mark.skipif(
    not (MODEL_DIR / "svm_model.pkl").exists()
    or not (MODEL_DIR / "svm_scaler.pkl").exists(),
    reason="SVM model/scaler not trained yet"
)
def test_svm_end_to_end_prediction():
    """SVM should produce a valid class index for a sine-wave input."""
    with open(MODEL_DIR / "svm_model.pkl", "rb") as f:
        svm = pickle.load(f)
    with open(MODEL_DIR / "svm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    feat   = extract_svm_features(_AUDIO).reshape(1, -1)
    feat   = scaler.transform(feat)
    pred   = svm.predict(feat)
    assert len(pred) == 1
    assert 0 <= pred[0] <= 7  # 8 emotion classes
