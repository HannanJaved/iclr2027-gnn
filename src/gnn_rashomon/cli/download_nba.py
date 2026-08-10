from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Social Power NBA dataset with kagglehub.")
    parser.add_argument("--dataset", default="noahgift/social-power-nba")
    parser.add_argument("--output-dir", default="data/NBA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise SystemExit("Install kagglehub before downloading NBA: python -m pip install kagglehub") from exc

    source_dir = Path(kagglehub.dataset_download(args.dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied.append(str(destination))
    print(f"Downloaded {args.dataset} to {source_dir}")
    print(f"Copied {len(copied)} files to {output_dir}")
    for path in copied:
        print(path)


if __name__ == "__main__":
    main()
