from __future__ import annotations

import argparse
from pathlib import Path


SUMMARY_PLOTS = [
    ("retained_count", "Retained Rashomon Models", "retained_count"),
    ("fraction_prediction_disagreement", "Prediction Disagreement Fraction", "disagreement"),
    ("predictive_entropy_mean", "Mean Predictive Entropy", "predictive_entropy"),
    ("variation_ratio_mean", "Mean Variation Ratio", "variation_ratio"),
    ("rashomon_capacity_mean", "Mean Rashomon Capacity", "rashomon_capacity"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregate report CSVs produced by build_report.")
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--top-k-correlations", type=int, default=18)
    return parser.parse_args()


def _label_rows(frame: object) -> list[str]:
    return [
        f"{row.dataset}\n{row.architecture}" if "architecture" in frame.columns else str(row.dataset)
        for row in frame.itertuples(index=False)
    ]


def _plot_summary(report_name: str, figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    path = figures_dir / f"{report_name}.multiplicity_summary.csv"
    if not path.exists():
        print(f"Skipping summary plots; missing {path}", flush=True)
        return
    frame = pd.read_csv(path).sort_values(["architecture", "dataset"])
    labels = _label_rows(frame)

    for column, ylabel, suffix in SUMMARY_PLOTS:
        if column not in frame.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(frame)), 4))
        ax.bar(labels, frame[column].astype(float))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = figures_dir / f"{report_name}.{suffix}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"Wrote {out}", flush=True)


def _plot_correlations(report_name: str, figures_dir: Path, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    path = figures_dir / f"{report_name}.structure_spearman.csv"
    if not path.exists():
        print(f"Skipping correlation heatmap; missing {path}", flush=True)
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    frame["label"] = frame["dataset"].astype(str)
    if "architecture" in frame.columns:
        frame["label"] = frame["label"] + "\n" + frame["architecture"].astype(str)
    frame["pair"] = frame["target"].astype(str) + " vs " + frame["feature"].astype(str)
    strongest = (
        frame.assign(abs_spearman=frame["spearman"].abs())
        .groupby("pair", as_index=False)["abs_spearman"]
        .max()
        .sort_values("abs_spearman", ascending=False)
        .head(top_k)["pair"]
    )
    subset = frame[frame["pair"].isin(strongest)]
    heatmap = subset.pivot_table(index="pair", columns="label", values="spearman", aggfunc="first")
    heatmap = heatmap.loc[list(strongest)]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(heatmap.columns)), max(4, 0.35 * len(heatmap))))
    image = ax.imshow(heatmap.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(heatmap.columns)), heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)), heatmap.index)
    ax.tick_params(axis="x", rotation=0)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    out = figures_dir / f"{report_name}.structure_spearman_heatmap.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_summary(args.report_name, figures_dir)
    _plot_correlations(args.report_name, figures_dir, args.top_k_correlations)


if __name__ == "__main__":
    main()
