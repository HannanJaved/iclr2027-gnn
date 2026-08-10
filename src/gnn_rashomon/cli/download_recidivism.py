from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ProPublica COMPAS recidivism CSV.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="data/Recidivism/compas-scores-two-years.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(args.url, output)
    (output.parent / "source.txt").write_text(f"url={args.url}\noutput={output}\n", encoding="utf-8")
    print(f"Downloaded recidivism CSV from {args.url} to {output}")


if __name__ == "__main__":
    main()
