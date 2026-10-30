from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


def evaluation_samples_match(
    records: Iterable[Mapping[str, object]],
    group_columns: Sequence[str],
    expected_groups: Iterable[tuple[object, ...]] | None = None,
) -> bool:
    """Return whether evaluated path/label identities match across all groups."""
    groups: dict[tuple[object, ...], set[tuple[str, str]]] = {}
    seen: dict[tuple[object, ...], set[tuple[str, str]]] = {}
    required = {*group_columns, "path", "y_true", "status"}

    for record in records:
        if not required.issubset(record):
            return False
        group = tuple(record[column] for column in group_columns)
        if any(value is None for value in group):
            return False
        identity = record.get("path")
        label = record.get("y_true")
        semantic_label = record.get("true_label", label)
        status = record.get("status")
        if not isinstance(identity, str) or not identity.strip():
            return False
        if isinstance(label, bool):
            return False
        try:
            numeric_label = int(label)
            if float(label) != numeric_label:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        if not isinstance(semantic_label, (str, int, float)) or not str(
            semantic_label
        ).strip():
            return False
        if status not in {"evaluated", "preprocessing_error", "inference_error"}:
            return False

        identity_key = (
            Path(identity).resolve().as_posix().casefold(),
            str(semantic_label),
        )
        current_seen = seen.setdefault(group, set())
        if identity_key in current_seen:
            return False
        current_seen.add(identity_key)
        if status == "evaluated":
            groups.setdefault(group, set()).add(identity_key)
        else:
            groups.setdefault(group, set())

    if expected_groups is not None:
        expected = {tuple(group) for group in expected_groups}
        if set(groups) != expected:
            return False
    identity_sets = list(groups.values())
    return len(identity_sets) > 1 and all(
        identities == identity_sets[0] for identities in identity_sets[1:]
    )


def classification_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
    }
