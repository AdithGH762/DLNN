from pathlib import Path

import pytest

from src.provenance import (
    create_run_provenance,
    sha256_file,
    validate_run_binding,
)


def test_content_fingerprint_is_stable_and_changes_with_file_contents(tmp_path: Path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"first model contents")
    first = sha256_file(artifact)
    assert sha256_file(artifact) == first

    artifact.write_bytes(b"updated model contents")

    assert sha256_file(artifact) != first


def test_run_id_is_recorded_with_input_fingerprints(tmp_path: Path):
    artifact = tmp_path / "manifest.csv"
    artifact.write_text("path,y_true\nrecording.wav,0\n", encoding="utf-8")

    metadata = create_run_provenance(
        evaluator="test",
        project_root=tmp_path,
        input_files={
            "manifest": {
                "path": artifact,
                "role": "evaluation_manifest",
                "required": True,
                "consumed_by_evaluator": True,
            }
        },
        settings={"conditions": ["clean"]},
        package_names=(),
    )

    assert metadata["run_id"]
    assert metadata["inputs"]["manifest"]["sha256"] == sha256_file(artifact)
    assert metadata["inputs"]["manifest"]["status"] == "fingerprinted"


def test_aggregate_evidence_and_metadata_must_share_run_id():
    assert validate_run_binding(["run-1", "run-1"], ["run-1"], "run-1") == "run-1"

    with pytest.raises(ValueError, match="do not share"):
        validate_run_binding(["run-1"], ["run-2"], "run-1")


def test_missing_fingerprint_is_recorded_instead_of_invented(tmp_path: Path):
    missing = tmp_path / "absent.keras"

    metadata = create_run_provenance(
        evaluator="test",
        project_root=tmp_path,
        input_files={
            "model": {
                "path": missing,
                "role": "inference_model",
                "required": True,
                "consumed_by_evaluator": True,
            }
        },
        settings={},
        package_names=(),
    )

    assert metadata["inputs"]["model"]["status"] == "missing"
    assert metadata["inputs"]["model"]["sha256"] is None
    assert metadata["missing_required_inputs"] == ["model"]
    assert not metadata["required_inputs_fingerprinted"]
