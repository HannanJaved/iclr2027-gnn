from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducibility manifest for saved artifacts."
    )
    parser.add_argument("--output", default="outputs/metadata/repro_manifest.json")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact path or glob pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--artifact-manifest",
        action="append",
        default=[],
        help=(
            "Existing reproducibility manifest whose recorded artifact paths should be "
            "rehash-listed in the new manifest. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--exclude-manifest-artifact-substring",
        action="append",
        default=[],
        help=(
            "Case-insensitive substring used to exclude paths imported from older manifests. "
            "Explicit --artifact paths are unaffected."
        ),
    )
    parser.add_argument(
        "--log-glob",
        action="append",
        default=[],
        help=(
            "Slurm log glob to record. Can be passed multiple times. "
            "Defaults to all .out/.err logs only when no explicit glob is supplied."
        ),
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[
            "gnn-rashomon",
            "torch",
            "torch-geometric",
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "networkx",
            "statsmodels",
            "matplotlib",
        ],
        help="Installed package name to record. Can be passed multiple times.",
    )
    return parser.parse_args()


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_patterns(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.update(Path(match) for match in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.add(path)
    return sorted(path for path in paths if path.is_file())


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
    }


def _paths_from_manifests(
    manifest_paths: list[str], exclude_substrings: list[str]
) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
    paths: set[Path] = set()
    sources: list[dict[str, Any]] = []
    missing: list[str] = []
    excludes = [value.lower() for value in exclude_substrings]
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources.append(_file_record(manifest_path))
        for record in payload.get("artifacts", []):
            artifact_path = Path(str(record["path"]))
            if any(value in str(artifact_path).lower() for value in excludes):
                continue
            if artifact_path.is_file():
                paths.add(artifact_path)
            else:
                missing.append(str(artifact_path))
    return sorted(paths), sources, sorted(set(missing))


def _package_versions(packages: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_manifest(
    output_path: Path,
    artifact_patterns: list[str],
    log_patterns: list[str],
    packages: list[str],
    artifact_manifests: list[str] | None = None,
    exclude_manifest_artifact_substrings: list[str] | None = None,
) -> dict[str, Any]:
    imported_paths, manifest_sources, missing_manifest_artifacts = _paths_from_manifests(
        artifact_manifests or [], exclude_manifest_artifact_substrings or []
    )
    artifact_paths = sorted(set(_expand_patterns(artifact_patterns)).union(imported_paths))
    artifacts = [_file_record(path) for path in artifact_paths]
    logs = [_file_record(path) for path in _expand_patterns(log_patterns)]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git": {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["branch", "--show-current"]),
            "status_short": _run_git(["status", "--short"]),
        },
        "platform": {
            "python": platform.python_version(),
            "executable": platform.python_implementation(),
            "system": platform.platform(),
        },
        "packages": _package_versions(packages),
        "artifact_manifest_sources": manifest_sources,
        "excluded_manifest_artifact_substrings": exclude_manifest_artifact_substrings or [],
        "missing_manifest_artifacts": missing_manifest_artifacts,
        "artifacts": artifacts,
        "logs": logs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    log_patterns = list(args.log_glob) or [".logs/*.out", ".logs/*.err"]
    manifest = build_manifest(
        output_path=Path(args.output),
        artifact_patterns=list(args.artifact),
        log_patterns=log_patterns,
        packages=list(dict.fromkeys(args.package)),
        artifact_manifests=list(args.artifact_manifest),
        exclude_manifest_artifact_substrings=list(args.exclude_manifest_artifact_substring),
    )
    print(
        "wrote reproducibility manifest "
        f"artifacts={len(manifest['artifacts'])} logs={len(manifest['logs'])} "
        f"path={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
