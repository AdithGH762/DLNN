from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np


def derive_noise_seed(recording_id: str | Path, condition: str, base_seed: int) -> int:
    """Derive a stable per-recording, per-condition seed from a base seed."""
    if not condition or not str(condition).strip():
        raise ValueError("Noise condition must be non-empty.")
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError("Base noise seed must be an integer.")
    normalized_id = Path(recording_id).resolve().as_posix().casefold()
    if not normalized_id:
        raise ValueError("Recording identity must be non-empty.")
    payload = f"{int(base_seed)}\0{normalized_id}\0{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def add_noise_for_condition(
    audio: np.ndarray,
    condition: str,
    snr_db: float | None,
    recording_id: str | Path,
    base_seed: int,
) -> tuple[np.ndarray, int | None]:
    """Apply deterministic noise for a noisy condition; leave clean audio unchanged."""
    if condition == "clean":
        if snr_db is not None:
            raise ValueError("Clean condition must not specify an SNR.")
        return audio, None
    if snr_db is None:
        raise ValueError(f"No SNR was provided for noisy condition {condition!r}.")
    seed = derive_noise_seed(recording_id, condition, base_seed)
    return add_white_noise_at_snr(audio, snr_db=snr_db, seed=seed), seed


def add_white_noise_at_snr(audio: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add white Gaussian noise at a target SNR. Returns float32 audio."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot add noise to empty audio.")
    signal_power = float(np.mean(x.astype(np.float64) ** 2))
    if signal_power < 1e-12:
        # SNR is undefined for silent signals; preserve silence rather than fabricate signal.
        return x.copy()
    noise = np.random.default_rng(seed).normal(0.0, 1.0, size=x.shape)
    noise_power = float(np.mean(noise ** 2))
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return (x + noise.astype(np.float32)).astype(np.float32)
