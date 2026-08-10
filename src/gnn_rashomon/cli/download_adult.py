from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Adult income dataset via KaggleHub.")
    parser.add_argument("--dataset", default="wenruliu/adult-income-dataset")
    parser.add_argument("--output", default="data/Adult/adult.csv")
    return parser.parse_args()


def _find_csv(root: Path) -> Path:
    candidates = sorted(root.rglob("*.csv"), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under KaggleHub dataset path: {root}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "kagglehub is not installed. Install it with `python -m pip install kagglehub` "
            "inside the active environment, then rerun this command."
        ) from exc

    downloaded = Path(kagglehub.dataset_download(args.dataset))
    source = _find_csv(downloaded)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    (output.parent / "kaggle_path.txt").write_text(
        f"dataset={args.dataset}\nsource={source}\noutput={output}\n",
        encoding="utf-8",
    )
    print(f"Downloaded KaggleHub dataset to {downloaded}")
    print(f"Copied Adult CSV from {source} to {output}")


if __name__ == "__main__":
    main()
