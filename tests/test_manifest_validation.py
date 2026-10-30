import pytest

from src.dataset import validate_test_manifest_labels


LABEL_MAPPING = {
    "0": "angry",
    "1": "calm",
    "2": "disgust",
    "3": "fearful",
    "4": "happy",
    "5": "neutral",
    "6": "sad",
    "7": "surprised",
}


def test_manifest_label_matches_filename_emotion():
    rows = [("03-01-05-01-02-01-12.wav", 0)]

    assert validate_test_manifest_labels(
        rows, LABEL_MAPPING, range(8), "test_manifest.csv"
    ) == [0]


def test_manifest_rejects_label_that_disagrees_with_filename():
    rows = [("03-01-05-01-02-01-12.wav", 1)]

    with pytest.raises(ValueError, match="label mismatch"):
        validate_test_manifest_labels(rows, LABEL_MAPPING, range(8), "test_manifest.csv")


def test_manifest_rejects_model_class_ids_that_do_not_match_mapping():
    rows = [("03-01-05-01-02-01-12.wav", 0)]

    with pytest.raises(ValueError, match="model class IDs"):
        validate_test_manifest_labels(rows, LABEL_MAPPING, range(7), "test_manifest.csv")


def test_manifest_rejects_duplicate_emotions_in_mapping():
    rows = [("03-01-05-01-02-01-12.wav", 0)]
    invalid_mapping = {**LABEL_MAPPING, "1": "angry"}

    with pytest.raises(ValueError, match="label mapping"):
        validate_test_manifest_labels(rows, invalid_mapping, range(8), "test_manifest.csv")
