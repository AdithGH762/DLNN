import numpy as np

from src.noise import add_noise_for_condition, derive_noise_seed


def test_same_recording_and_condition_produce_same_seed_and_noise():
    audio = np.linspace(-0.5, 0.5, 256, dtype=np.float32)

    seed_a = derive_noise_seed("recordings/actor_01.wav", "snr_20db", 2000)
    seed_b = derive_noise_seed("recordings/actor_01.wav", "snr_20db", 2000)
    noisy_a, applied_seed_a = add_noise_for_condition(
        audio, "snr_20db", 20, "recordings/actor_01.wav", 2000
    )
    noisy_b, applied_seed_b = add_noise_for_condition(
        audio, "snr_20db", 20, "recordings/actor_01.wav", 2000
    )

    assert seed_a == seed_b == applied_seed_a == applied_seed_b
    np.testing.assert_array_equal(noisy_a, noisy_b)


def test_manifest_reordering_does_not_change_recording_seeds():
    recording_ids = ["recordings/actor_01.wav", "recordings/actor_02.wav"]
    original = {
        recording_id: derive_noise_seed(recording_id, "snr_10db", 1000)
        for recording_id in recording_ids
    }
    reordered = {
        recording_id: derive_noise_seed(recording_id, "snr_10db", 1000)
        for recording_id in reversed(recording_ids)
    }

    assert original == reordered


def test_noise_conditions_have_distinct_deterministic_streams():
    seeds = {
        derive_noise_seed("recordings/actor_01.wav", condition, 2000)
        for condition in ("snr_20db", "snr_10db", "snr_0db")
    }

    assert len(seeds) == 3


def test_clean_condition_does_not_add_noise_or_use_a_seed():
    audio = np.linspace(-0.5, 0.5, 256, dtype=np.float32)

    processed, seed = add_noise_for_condition(
        audio, "clean", None, "recordings/actor_01.wav", 2000
    )

    assert seed is None
    np.testing.assert_array_equal(processed, audio)
