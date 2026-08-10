from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UCI Statlog German Credit data.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="data/GermanCredit/german.data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(args.url, output)
    (output.parent / "source.txt").write_text(f"url={args.url}\noutput={output}\n", encoding="utf-8")
    print(f"Downloaded German Credit data from {args.url} to {output}", flush=True)


if __name__ == "__main__":
    main()
