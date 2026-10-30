import numpy as np
from src.noise import add_white_noise_at_snr


def test_noise_shape_and_snr():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, 16000).astype(np.float32)
    y = add_white_noise_at_snr(x, 10, seed=3)
    assert y.shape == x.shape
    noise = y - x
    measured = 10 * np.log10(np.mean(x.astype(float)**2) / np.mean(noise.astype(float)**2))
    assert abs(measured - 10) < 0.2


def test_silent_audio_remains_silent():
    x = np.zeros(1000, dtype=np.float32)
    assert np.allclose(add_white_noise_at_snr(x, 0), x)
