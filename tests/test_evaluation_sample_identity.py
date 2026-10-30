from src.metrics import evaluation_samples_match


def _record(condition, path, label=0, status="evaluated"):
    return {
        "condition": condition,
        "path": path,
        "y_true": label,
        "y_pred": label if status == "evaluated" else None,
        "status": status,
        "error": "",
    }


def test_identical_recordings_and_labels_match_across_conditions():
    records = [
        _record("clean", "dataset/recording_1.wav"),
        _record("clean", "dataset/recording_2.wav", label=1),
        _record("20db", "dataset/recording_1.wav"),
        _record("20db", "dataset/recording_2.wav", label=1),
    ]

    assert evaluation_samples_match(records, ["condition"])


def test_equal_counts_with_different_recordings_do_not_match():
    records = [
        _record("clean", "dataset/recording_1.wav"),
        _record("20db", "dataset/recording_2.wav"),
    ]

    assert not evaluation_samples_match(records, ["condition"])


def test_same_recording_with_different_true_labels_does_not_match():
    records = [
        _record("clean", "dataset/recording_1.wav", label=0),
        _record("20db", "dataset/recording_1.wav", label=1),
    ]

    assert not evaluation_samples_match(records, ["condition"])


def test_recording_failed_in_one_condition_does_not_match():
    records = [
        _record("clean", "dataset/recording_1.wav"),
        _record("clean", "dataset/recording_2.wav"),
        _record("20db", "dataset/recording_1.wav"),
        _record(
            "20db",
            "dataset/recording_2.wav",
            status="preprocessing_error",
        ),
    ]

    assert not evaluation_samples_match(records, ["condition"])


def test_model_specific_class_ids_match_by_semantic_true_label():
    records = [
        {
            **_record("clean", "dataset/recording_1.wav", label=0),
            "model_key": "original",
            "true_label": "calm",
        },
        {
            **_record("clean", "dataset/recording_1.wav", label=2),
            "model_key": "improved",
            "true_label": "calm",
        },
    ]

    assert evaluation_samples_match(
        records,
        ["model_key", "condition"],
        expected_groups=[("original", "clean"), ("improved", "clean")],
    )
