from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import pandas as pd

from config import DATA_DIR, EMOTION_CODES, SPEECH_MODALITY

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


def parse_ravdess_filename(path: Path) -> dict | None:
    """Parse the standard RAVDESS filename, e.g. 03-01-05-01-02-01-12.wav.

    Returns None for:
    - Filenames that do not match the 7-part RAVDESS convention.
    - Non-speech modalities (songs, video-only).  Only modality 03
      (audio-only speech) is accepted.
    - Unknown emotion codes.
    """
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    # Enforce speech-only modality.
    if parts[0] != SPEECH_MODALITY:
        return None
    if parts[2] not in EMOTION_CODES:
        return None
    try:
        actor_id = int(parts[6])
    except ValueError:
        return None
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion_code": parts[2],
        "emotion": EMOTION_CODES[parts[2]],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor_id": actor_id,
    }


def validate_test_manifest_labels(
    rows: Iterable[tuple[object, object]],
    mapping: Mapping[str, str],
    model_class_ids: Iterable[object],
    manifest_name: str,
) -> list[int]:
    """Validate manifest labels against RAVDESS filenames and model class IDs."""
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(f"{manifest_name}: label mapping must be a non-empty object.")
    expected_mapping_keys = {str(index) for index in range(len(mapping))}
    if (
        set(mapping) != expected_mapping_keys
        or any(
            not isinstance(label, str) or label not in EMOTION_CODES.values()
            for label in mapping.values()
        )
        or len(set(mapping.values())) != len(mapping)
    ):
        raise ValueError(
            f"{manifest_name}: label mapping must have contiguous zero-based IDs "
            "and unique recognized RAVDESS emotion labels."
        )

    try:
        normalized_class_ids = []
        for class_id in model_class_ids:
            normalized = int(class_id)
            if isinstance(class_id, bool) or normalized != class_id:
                raise ValueError
            normalized_class_ids.append(normalized)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{manifest_name}: model contains an invalid class ID.") from exc
    expected_class_ids = set(range(len(mapping)))
    if (
        len(normalized_class_ids) != len(set(normalized_class_ids))
        or set(normalized_class_ids) != expected_class_ids
    ):
        raise ValueError(
            f"{manifest_name}: model class IDs must match the contiguous IDs "
            "in its label mapping."
        )

    emotion_to_id = {emotion: int(class_id) for class_id, emotion in mapping.items()}
    validated_labels = []
    for row_number, (raw_path, raw_label) in enumerate(rows, start=2):
        if (
            raw_path is None
            or pd.isna(raw_path)
            or raw_label is None
            or pd.isna(raw_label)
        ):
            raise ValueError(
                f"{manifest_name}: row {row_number} has an empty path or y_true label."
            )
        parsed = parse_ravdess_filename(Path(str(raw_path)))
        if parsed is None:
            raise ValueError(
                f"{manifest_name}: row {row_number} has a path that does not encode "
                "a recognized RAVDESS speech emotion."
            )
        try:
            label = int(raw_label)
            if isinstance(raw_label, bool) or float(raw_label) != label:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{manifest_name}: row {row_number} has an invalid integer y_true label."
            ) from exc

        expected_label = emotion_to_id.get(parsed["emotion"])
        if expected_label is None or label != expected_label:
            expected_name = parsed["emotion"]
            raise ValueError(
                f"{manifest_name}: row {row_number} label mismatch for "
                f"{Path(str(raw_path)).name}: y_true={label}, but filename emotion "
                f"'{expected_name}' maps to class ID {expected_label!r}."
            )
        validated_labels.append(label)
    return validated_labels


def discover_ravdess(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Discover RAVDESS speech-only files recursively and return parsed metadata.

    Only files matching the 7-part filename convention AND having modality 03
    (audio-only speech) are included.  Song and video files are silently skipped.
    """
    data_dir = Path(data_dir)
    rows = []
    if not data_dir.exists():
        return pd.DataFrame()
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            row = parse_ravdess_filename(path)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def dataset_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(columns=["emotion", "files"]), pd.DataFrame(columns=["actor_id", "files"])
    by_emotion = df.groupby("emotion").size().rename("files").reset_index().sort_values("emotion")
    by_actor = df.groupby("actor_id").size().rename("files").reset_index().sort_values("actor_id")
    return by_emotion, by_actor
