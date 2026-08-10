from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".json", ".pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare analysis artifacts produced by two software environments."
    )
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--patterns", nargs="+", default=["**/*.csv", "**/*.json", "**/*.pt"])
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _compare_values(reference: Any, candidate: Any, rtol: float, atol: float) -> bool:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        return reference.keys() == candidate.keys() and all(
            _compare_values(reference[key], candidate[key], rtol, atol) for key in reference
        )
    if isinstance(reference, list) and isinstance(candidate, list):
        return len(reference) == len(candidate) and all(
            _compare_values(left, right, rtol, atol)
            for left, right in zip(reference, candidate, strict=True)
        )
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        return bool(np.isclose(reference, candidate, rtol=rtol, atol=atol, equal_nan=True))
    return reference == candidate


def _compare_csv(reference: Path, candidate: Path, rtol: float, atol: float) -> tuple[bool, str]:
    left = pd.read_csv(reference)
    right = pd.read_csv(candidate)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        detail = (
            f"schema mismatch: {left.shape}/{list(left.columns)} "
            f"vs {right.shape}/{list(right.columns)}"
        )
        return False, detail
    for column in left.columns:
        left_numeric = pd.api.types.is_numeric_dtype(left[column])
        right_numeric = pd.api.types.is_numeric_dtype(right[column])
        if left_numeric and right_numeric:
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                return False, f"numeric mismatch in column {column}"
        elif not left[column].fillna("<NA>").equals(right[column].fillna("<NA>")):
            return False, f"value mismatch in column {column}"
    return True, "equivalent"


def _compare_pt(reference: Path, candidate: Path, rtol: float, atol: float) -> tuple[bool, str]:
    import torch

    left = torch.load(reference, map_location="cpu", weights_only=True)
    right = torch.load(candidate, map_location="cpu", weights_only=True)
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False, "only tensor .pt artifacts are supported"
    if left.shape != right.shape:
        return False, f"shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}"
    return (
        (True, "equivalent")
        if torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
        else (False, "tensor values differ")
    )


def compare_file(reference: Path, candidate: Path, rtol: float, atol: float) -> tuple[bool, str]:
    if reference.suffix == ".csv":
        return _compare_csv(reference, candidate, rtol, atol)
    if reference.suffix == ".json":
        left = json.loads(reference.read_text(encoding="utf-8"))
        right = json.loads(candidate.read_text(encoding="utf-8"))
        equivalent = _compare_values(left, right, rtol, atol)
        return equivalent, "equivalent" if equivalent else "JSON values differ"
    if reference.suffix == ".pt":
        return _compare_pt(reference, candidate, rtol, atol)
    return False, f"unsupported suffix {reference.suffix}"


def build_report(
    reference_root: Path,
    candidate_root: Path,
    patterns: list[str],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    relative_paths = sorted(
        {
            path.relative_to(reference_root)
            for pattern in patterns
            for path in reference_root.glob(pattern)
            if path.is_file() and path.suffix in SUPPORTED_SUFFIXES
        },
        key=str,
    )
    rows = []
    for relative in relative_paths:
        reference = reference_root / relative
        candidate = candidate_root / relative
        if not candidate.exists():
            equivalent, detail = False, "missing candidate"
        else:
            equivalent, detail = compare_file(reference, candidate, rtol, atol)
        rows.append({"path": str(relative), "equivalent": equivalent, "detail": detail})
    passed = bool(rows) and all(row["equivalent"] for row in rows)
    return {
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "rtol": rtol,
        "atol": atol,
        "artifact_count": len(rows),
        "equivalent_count": sum(int(row["equivalent"]) for row in rows),
        "passed": passed,
        "artifacts": rows,
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        Path(args.reference_root),
        Path(args.candidate_root),
        list(args.patterns),
        args.rtol,
        args.atol,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote environment-equivalence report to {output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
