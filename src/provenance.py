from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path


def input_key(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        identity = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        identity = str(resolved)
    return f"recording:{identity}"


def sha256_file(path: Path) -> str:
    """Return a content fingerprint without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_input(
    path: Path,
    project_root: Path,
    *,
    role: str,
    required: bool,
    consumed_by_evaluator: bool,
) -> dict:
    path = path.resolve()
    project_root = project_root.resolve()
    try:
        display_path = path.relative_to(project_root).as_posix()
    except ValueError:
        display_path = str(path)
    if path.is_file():
        return {
            "role": role,
            "required": required,
            "consumed_by_evaluator": consumed_by_evaluator,
            "path": display_path,
            "status": "fingerprinted",
            "sha256": sha256_file(path),
        }
    return {
        "role": role,
        "required": required,
        "consumed_by_evaluator": consumed_by_evaluator,
        "path": display_path,
        "status": "missing",
        "sha256": None,
    }


def register_run_input(
    provenance: dict,
    name: str,
    path: Path,
    project_root: Path,
    *,
    role: str,
    required: bool,
    consumed_by_evaluator: bool,
) -> dict:
    """Add a runtime-discovered input and maintain missing-input summaries."""
    descriptor = fingerprint_input(
        path,
        project_root,
        role=role,
        required=required,
        consumed_by_evaluator=consumed_by_evaluator,
    )
    provenance["inputs"][name] = descriptor
    missing_key = (
        "missing_required_inputs" if required else "missing_context_inputs"
    )
    other_key = (
        "missing_context_inputs" if required else "missing_required_inputs"
    )
    if descriptor["status"] == "missing":
        if name not in provenance[missing_key]:
            provenance[missing_key].append(name)
    else:
        provenance[missing_key] = [
            item for item in provenance[missing_key] if item != name
        ]
    provenance[other_key] = [
        item for item in provenance[other_key] if item != name
    ]
    provenance["required_inputs_fingerprinted"] = not provenance[
        "missing_required_inputs"
    ]
    return descriptor


def _git_revision(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def create_run_provenance(
    *,
    evaluator: str,
    project_root: Path,
    input_files: Mapping[str, Mapping[str, object]],
    settings: Mapping[str, object],
    package_names: Iterable[str],
) -> dict:
    """Create run metadata, hashing each existing input from its contents."""
    project_root = project_root.resolve()
    inputs = {}
    missing_required = []
    missing_context = []
    for name, descriptor in input_files.items():
        path_value = descriptor.get("path")
        path = Path(path_value) if path_value is not None else None
        item = (
            fingerprint_input(
                path,
                project_root,
                role=str(descriptor["role"]),
                required=bool(descriptor.get("required", False)),
                consumed_by_evaluator=bool(
                    descriptor.get("consumed_by_evaluator", False)
                ),
            )
            if path is not None
            else {
                "role": str(descriptor["role"]),
                "required": bool(descriptor.get("required", False)),
                "consumed_by_evaluator": bool(
                    descriptor.get("consumed_by_evaluator", False)
                ),
                "path": None,
                "status": "missing",
                "sha256": None,
            }
        )
        if item["status"] == "missing":
            (missing_required if item["required"] else missing_context).append(name)
        inputs[name] = item

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in package_names:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "evaluator": evaluator,
        "code_revision": _git_revision(project_root),
        "software_versions": versions,
        "settings": dict(settings),
        "inputs": inputs,
        "missing_required_inputs": missing_required,
        "missing_context_inputs": missing_context,
        "required_inputs_fingerprinted": not missing_required,
    }


def run_provenance_path(output_dir: Path, evaluator: str, run_id: str) -> Path:
    return output_dir / f"{evaluator}_run_{run_id}.json"


def write_run_provenance(path: Path, provenance: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finish_run_provenance(path: Path) -> dict:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["status"] = "completed"
    provenance["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_run_provenance(path, provenance)
    return provenance


def validate_run_binding(
    aggregate_run_ids: Iterable[object],
    evidence_run_ids: Iterable[object],
    metadata_run_id: object,
) -> str:
    """Require aggregate rows, per-record evidence, and metadata to share one ID."""
    aggregate_ids = {str(value).strip() for value in aggregate_run_ids}
    evidence_ids = {str(value).strip() for value in evidence_run_ids}
    metadata_id = str(metadata_run_id).strip()
    if (
        len(aggregate_ids) != 1
        or "" in aggregate_ids
        or evidence_ids != aggregate_ids
        or metadata_id not in aggregate_ids
    ):
        raise ValueError(
            "Aggregate results, per-record evidence, and run metadata do not share "
            "one evaluation run ID."
        )
    return metadata_id
