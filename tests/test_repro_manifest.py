import json
import sys

from gnn_rashomon.cli.build_repro_manifest import build_manifest, parse_args


def test_build_repro_manifest_records_hashes_and_package_versions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "outputs" / "figures" / "table.csv"
    log = tmp_path / ".logs" / "job.out"
    artifact.parent.mkdir(parents=True)
    log.parent.mkdir(parents=True)
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    log.write_text("done\n", encoding="utf-8")

    output = tmp_path / "outputs" / "metadata" / "manifest.json"
    manifest = build_manifest(
        output_path=output,
        artifact_patterns=["outputs/figures/*.csv"],
        log_patterns=[".logs/*.out"],
        packages=["json", "definitely-not-installed-package"],
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == manifest
    assert len(saved["artifacts"]) == 1
    assert len(saved["logs"]) == 1
    assert saved["artifacts"][0]["sha256"]
    assert saved["packages"]["definitely-not-installed-package"] is None


def test_explicit_log_globs_do_not_include_default_patterns(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_repro_manifest", "--log-glob", ".logs/focal_*"],
    )

    args = parse_args()

    assert args.log_glob == [".logs/focal_*"]


def test_build_manifest_imports_paths_and_excludes_withdrawn_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    retained = tmp_path / "outputs" / "metrics" / "cora.csv"
    withdrawn = tmp_path / "outputs" / "metrics" / "adult-old.csv"
    corrected = tmp_path / "outputs" / "release" / "adult-corrected.csv"
    for path in [retained, withdrawn, corrected]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value\n1\n", encoding="utf-8")

    source = tmp_path / "source_manifest.json"
    source.write_text(
        json.dumps(
            {"artifacts": [{"path": str(retained)}, {"path": str(withdrawn)}]}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "release_manifest.json"
    manifest = build_manifest(
        output_path=output,
        artifact_patterns=[str(corrected)],
        log_patterns=[],
        packages=[],
        artifact_manifests=[str(source)],
        exclude_manifest_artifact_substrings=["adult"],
    )

    paths = {record["path"] for record in manifest["artifacts"]}
    assert str(retained) in paths
    assert str(corrected) in paths
    assert str(withdrawn) not in paths
    assert manifest["missing_manifest_artifacts"] == []
    assert manifest["artifact_manifest_sources"][0]["sha256"]
