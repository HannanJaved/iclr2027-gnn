#!/usr/bin/env python3
"""Generate camera-ready paper figures from released experiment artifacts.

Recommended main-text set:
  Fig 1  Structural correlates (homophily / entropy vs diameter)
  Fig 2  Fixed-ensemble random rewiring dose response
  Fig 3  Local matched treatment–control diameter effects
  Fig 4  Explanation stability: multiplicity vs degree
  Fig 5  NBA accuracy–fairness Pareto with FairGNN / FairSIN
  Fig 6  Validation-accuracy sensitivity of disagreement

Usage (from repo root, with project venv):
  .venv/bin/python scripts/paper_figures/generate_paper_figures.py
  .venv/bin/python scripts/paper_figures/generate_paper_figures.py --output-dir outputs/paper_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
DATASET_ORDER = ["citeseer", "cora", "pubmed"]
DATASET_LABELS = {"citeseer": "CiteSeer", "cora": "Cora", "pubmed": "PubMed"}
# Okabe–Ito colorblind-safe palette (https://jfly.uni-koeln.de/color/)
PALETTE = {
    "citeseer": "#0072B2",  # blue
    "cora": "#E69F00",  # orange
    "pubmed": "#009E73",  # bluish green
    "gat": "#CC79A7",  # reddish purple
    "fairgnn": "#D55E00",  # vermillion
    "fairsin": "#56B4E9",  # sky blue
    "rashomon": "#000000",  # black
    "degree": "#0072B2",  # blue
    "diameter": "#D55E00",  # vermillion
    "neutral": "#999999",
    "retained": "#000000",  # black (Okabe–Ito)
    "val05": "#E69F00",  # orange
    "val02": "#0072B2",  # blue
    "train_loss": "#E69F00",  # orange/yellow (Okabe–Ito)
    "joint": "#0072B2",  # blue
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_figures"),
        help="Directory for PDF/PNG figures and companion CSVs.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
    extra_artists: list | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.15,
            bbox_extra_artists=extra_artists or (),
        )
        paths.append(path)
        print(f"Wrote {path}")
    plt.close(fig)
    return paths


def _diameter_column(frame: pd.DataFrame) -> str:
    if "rashomon_capacity" in frame.columns:
        return "rashomon_capacity"
    if "probability_diameter" in frame.columns:
        return "probability_diameter"
    raise KeyError("No diameter column found")


def fig1_structural_correlates(root: Path, output_dir: Path, dpi: int) -> None:
    """Observational structural correlates for citation GCN seed sets."""
    spearman = pd.read_csv(root / "outputs/figures/seed_relative0.1.structure_spearman.csv")
    focus_features = ["local_homophily", "neighborhood_label_entropy", "degree"]
    focus = spearman[
        (spearman["dataset"].isin(DATASET_ORDER))
        & (spearman["target"] == "rashomon_capacity")
        & (spearman["feature"].isin(focus_features))
    ].copy()
    focus["feature_label"] = focus["feature"].map(
        {
            "local_homophily": "Local homophily",
            "neighborhood_label_entropy": "Neighborhood label entropy",
            "degree": "Degree",
        }
    )
    focus.to_csv(output_dir / "fig1_structural_correlates_spearman.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.05, 1.2]})

    # Panel A: Spearman bars
    ax = axes[0]
    features = ["Local homophily", "Neighborhood label entropy", "Degree"]
    short_features = ["Homophily", "Label entropy", "Degree"]
    x = np.arange(len(features))
    width = 0.25
    for i, dataset in enumerate(DATASET_ORDER):
        vals = [
            float(
                focus.loc[
                    (focus["dataset"] == dataset) & (focus["feature_label"] == feature),
                    "spearman",
                ].iloc[0]
            )
            for feature in features
        ]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=PALETTE[dataset],
            label=DATASET_LABELS[dataset],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-0.32, 0.32)
    ax.set_xticks(x, short_features)
    ax.set_ylabel(r"Spearman $\rho$ with probability diameter")
    ax.set_title("A. Citation-graph correlates")
    # Homophily bars are negative, so the upper-left quadrant stays clear.
    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        loc="upper left",
        handlelength=1.4,
        borderpad=0.35,
    )

    # Panel B: Cora scatter (binned means for readability)
    nodes = pd.read_csv(
        root / "outputs/metrics/cora-gcn-seed-relative0.1-epochs200.structure_nodes.csv"
    )
    diam = _diameter_column(nodes)
    ax = axes[1]
    ax.scatter(
        nodes["local_homophily"],
        nodes[diam],
        s=6,
        alpha=0.18,
        color=PALETTE["cora"],
        rasterized=True,
        label="Nodes",
    )
    bins = pd.qcut(nodes["local_homophily"], q=10, duplicates="drop")
    grouped = nodes.groupby(bins, observed=True).agg(
        x=("local_homophily", "mean"),
        y=(diam, "mean"),
        y_lo=(diam, lambda s: float(np.quantile(s, 0.25))),
        y_hi=(diam, lambda s: float(np.quantile(s, 0.75))),
    )
    ax.errorbar(
        grouped["x"],
        grouped["y"],
        yerr=[grouped["y"] - grouped["y_lo"], grouped["y_hi"] - grouped["y"]],
        fmt="o-",
        color="black",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label="Decile mean ± IQR",
    )
    ax.set_xlabel("Local homophily")
    ax.set_ylabel("Probability diameter")
    ax.set_title("B. Cora GCN seed set")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    save_fig(fig, output_dir, "fig1_structural_correlates", dpi)


def fig2_rewiring_dose_response(root: Path, output_dir: Path, dpi: int) -> None:
    """Random rewiring dose response for mean probability diameter.

    Shaded region marks the accuracy-invalid large-perturbation regime where
    retained models leave the δ=0.02 validation band on G' (K_δ(G')≈0).
    """
    gcn = pd.read_csv(root / "outputs/repeated_rewiring/repeated_rewiring_inference.csv")
    gat = pd.read_csv(
        root / "outputs/non_gcn_rewiring/pubmed_gat/pubmed_gat_repeated_rewiring_inference.csv"
    )
    effect = "delta_mean_probability_diameter"
    random_gcn = gcn[(gcn["mode"] == "random") & (gcn["effect"] == effect)].copy()
    random_gat = gat[(gat["mode"] == "random") & (gat["effect"] == effect)].copy()
    random_gcn["architecture"] = "GCN"
    random_gat["architecture"] = "GAT"
    random_gat["dataset"] = "pubmed"
    out = pd.concat([random_gcn, random_gat], ignore_index=True)
    out.to_csv(output_dir / "fig2_rewiring_dose_response.csv", index=False)

    # Audited: random 0.01 keeps most models near-optimal. Strengths ≥0.10
    # have K_δ(G')=0 on CiteSeer/Cora GCN and about one viable model on
    # PubMed (Appendix Table accuracy_under_rewiring). Strength 0.05 is a
    # transition regime (partial PubMed viability) and is left unshaded.
    invalid_from = 0.10

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    xmax = float(out["strength"].max()) + 0.02
    ax.axvspan(
        invalid_from,
        xmax,
        color="#E69F00",
        alpha=0.15,
        zorder=0,
        label=r"accuracy-invalid ($K_{\delta}\!\approx\!0$ at $\geq0.10$)",
    )
    ax.axvline(invalid_from, color="#E69F00", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    for dataset in DATASET_ORDER:
        sub = random_gcn[random_gcn["dataset"] == dataset].sort_values("strength")
        ax.errorbar(
            sub["strength"],
            sub["mean_effect"],
            yerr=[
                sub["mean_effect"] - sub["bootstrap_ci_low"],
                sub["bootstrap_ci_high"] - sub["mean_effect"],
            ],
            fmt="o-",
            color=PALETTE[dataset],
            label=f"{DATASET_LABELS[dataset]} GCN",
            markersize=5,
            capsize=2,
            linewidth=1.5,
            zorder=3,
        )
    gat_sorted = random_gat.sort_values("strength")
    ax.errorbar(
        gat_sorted["strength"],
        gat_sorted["mean_effect"],
        yerr=[
            gat_sorted["mean_effect"] - gat_sorted["bootstrap_ci_low"],
            gat_sorted["bootstrap_ci_high"] - gat_sorted["mean_effect"],
        ],
        fmt="s--",
        color=PALETTE["gat"],
        label="PubMed GAT",
        markersize=5,
        capsize=2,
        linewidth=1.5,
        zorder=3,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("Random rewiring strength")
    ax.set_ylabel(r"$\Delta$ mean probability diameter")
    ax.set_title("Fixed-ensemble random rewiring")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_fig(fig, output_dir, "fig2_rewiring_dose_response", dpi)


def fig7_relative_dose_response(root: Path, output_dir: Path, dpi: int) -> None:
    """Relative vs absolute dose response with accuracy-invalid shading."""
    csv_path = output_dir / "fig7_relative_dose_response.csv"
    if not csv_path.exists():
        csv_path = root / "outputs/paper_figures/fig7_relative_dose_response.csv"
    frame = pd.read_csv(csv_path)
    # Strengths ≥0.10 are accuracy-invalid for CiteSeer/Cora and nearly so
    # for PubMed; do not shade the 0.05 transition regime.
    invalid_from = 0.10

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    xmax = float(frame["strength"].max()) + 0.02
    for ax, ycol, ylabel in (
        (axes[0], "mean_relative_delta_D", r"Relative $\Delta D$ (%)"),
        (axes[1], "mean_delta_D", r"Absolute $\Delta D$"),
    ):
        ax.axvspan(invalid_from, xmax, color="#E69F00", alpha=0.15, zorder=0)
        ax.axvline(invalid_from, color="#E69F00", linestyle="--", linewidth=1.0, alpha=0.8)
        for dataset in DATASET_ORDER:
            sub = frame[frame["dataset"] == dataset].sort_values("strength")
            y = sub[ycol] * (100.0 if ycol.startswith("mean_relative") else 1.0)
            ax.plot(
                sub["strength"],
                y,
                "o-",
                color=PALETTE[dataset],
                label=DATASET_LABELS[dataset],
                markersize=5,
                linewidth=1.5,
            )
        ax.set_xlim(0.0, xmax)
        ax.set_xlabel("Rewiring strength")
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, color="black", linewidth=0.8)
    axes[0].text(
        0.98,
        0.95,
        r"$K_{\delta}(G')\approx 0$ at $\geq 0.10$",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color="#D55E00",
        fontsize=8,
    )
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        r"Random-rewiring dose response (shaded: $K_{\delta}\approx 0$ at strength $\geq 0.10$)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig7_relative_dose_response", dpi)


def fig3_local_matched_effects(root: Path, output_dir: Path, dpi: int) -> None:
    """Forest plot of matched local-entanglement treatment–control effects."""
    gcn = pd.read_csv(root / "outputs/repeated_rewiring/repeated_rewiring_inference.csv")
    gat = pd.read_csv(
        root / "outputs/non_gcn_rewiring/pubmed_gat/pubmed_gat_repeated_rewiring_inference.csv"
    )
    effect = "matched_treatment_control_delta_diameter"
    rows = []
    for dataset in DATASET_ORDER:
        sub = gcn[(gcn["dataset"] == dataset) & (gcn["effect"] == effect)].iloc[0]
        rows.append(
            {
                "label": f"{DATASET_LABELS[dataset]} GCN",
                "dataset": dataset,
                "architecture": "GCN",
                "mean_effect": float(sub["mean_effect"]),
                "ci_low": float(sub["bootstrap_ci_low"]),
                "ci_high": float(sub["bootstrap_ci_high"]),
                "positive_count": int(sub["positive_effect_count"]),
                "realization_count": int(sub["realization_count"]),
            }
        )
    gat_row = gat[gat["effect"] == effect].iloc[0]
    rows.append(
        {
            "label": "PubMed GAT",
            "dataset": "pubmed",
            "architecture": "GAT",
            "mean_effect": float(gat_row["mean_effect"]),
            "ci_low": float(gat_row["bootstrap_ci_low"]),
            "ci_high": float(gat_row["bootstrap_ci_high"]),
            "positive_count": int(gat_row["positive_effect_count"]),
            "realization_count": int(gat_row["realization_count"]),
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "fig3_local_matched_effects.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    y = np.arange(len(frame))[::-1]
    colors = [
        PALETTE["gat"] if row.architecture == "GAT" else PALETTE[row.dataset]
        for row in frame.itertuples(index=False)
    ]
    for idx, (yi, row) in enumerate(zip(y, frame.itertuples(index=False), strict=True)):
        ax.errorbar(
            row.mean_effect,
            yi,
            xerr=[[row.mean_effect - row.ci_low], [row.ci_high - row.mean_effect]],
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            elinewidth=2.0,
            capsize=3,
            markersize=6,
        )
        ax.text(
            row.ci_high + 0.0004,
            yi,
            f"{row.positive_count}/{row.realization_count} > 0",
            va="center",
            fontsize=8,
        )
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y, frame["label"])
    ax.set_xlabel(r"Matched treatment − control $\Delta$ diameter")
    ax.set_title("Local entanglement intervention")
    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_local_matched_effects", dpi)


def _collect_explanation_regressions(root: Path) -> pd.DataFrame:
    rows = []
    paths = sorted((root / "outputs/explanations").glob("*.regression.csv"))
    skip_tokens = ("smoke", "tiny")
    for path in paths:
        name = path.name
        if any(token in name for token in skip_tokens):
            continue
        if not name.endswith(".regression.csv"):
            continue
        stem = name[: -len(".regression.csv")]
        # Method is the final dotted token; set ids may contain decimals like relative0.1.
        known_methods = ("gnnexplainer", "pgexplainer", "gat_attention")
        method = None
        set_id = None
        for candidate in known_methods:
            suffix = f".{candidate}"
            if stem.endswith(suffix):
                method = candidate
                set_id = stem[: -len(suffix)]
                break
        if method is None or set_id is None:
            # Legacy untagged regression file; skip.
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty or "outcome" not in frame.columns:
            continue
        frame = frame[frame["outcome"] == "top_k_jaccard_mean"].copy()
        if frame.empty:
            continue
        dataset = set_id.split("-")[0]
        architecture = set_id.split("-")[1]
        for row in frame.itertuples(index=False):
            if row.term in {"Intercept"}:
                continue
            rows.append(
                {
                    "set_id": set_id,
                    "dataset": dataset,
                    "architecture": architecture,
                    "method": method,
                    "term": row.term,
                    "coefficient": float(row.coefficient),
                    "std_error": float(row.std_error),
                    "p_value": float(row.p_value),
                    "n": int(row.n),
                    "path": str(path.relative_to(root)),
                }
            )
    return pd.DataFrame(rows)


def fig4_explanation_stability(root: Path, output_dir: Path, dpi: int) -> None:
    """Multiplicity vs degree coefficients for explanation Jaccard stability."""
    regs = _collect_explanation_regressions(root)
    focus_methods = {"gnnexplainer", "pgexplainer", "gat_attention"}
    regs = regs[regs["method"].isin(focus_methods)].copy()
    # Main-text panel: GNNExplainer diameter and degree across healthy sets
    gnn = regs[
        (regs["method"] == "gnnexplainer") & (regs["term"].isin(["rashomon_capacity", "degree"]))
    ].copy()
    gnn.to_csv(output_dir / "fig4_explanation_regressions.csv", index=False)
    regs.to_csv(output_dir / "fig4_explanation_regressions_all.csv", index=False)

    # Stable display order
    order = [
        "cora-gcn-seed-relative0.1-epochs200",
        "cora-gat-canonical_seed-relative2-epochs200-valaccdelta0p02",
        "cora-appnp-arch_appnp-relative2-epochs200",
        "pubmed-gat-arch_gat-relative2-epochs200",
        "pubmed-appnp-arch_appnp-relative2-epochs200",
        "amazon_photo-gcn-seed-relative0.1-epochs200",
        "amazon_photo-gat-arch_gat-relative0.1-epochs200",
        "amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200",
    ]
    labels = {
        "cora-gcn-seed-relative0.1-epochs200": "Cora GCN",
        "cora-gat-canonical_seed-relative2-epochs200-valaccdelta0p02": "Cora GAT",
        "cora-appnp-arch_appnp-relative2-epochs200": "Cora APPNP",
        "pubmed-gat-arch_gat-relative2-epochs200": "PubMed GAT",
        "pubmed-appnp-arch_appnp-relative2-epochs200": "PubMed APPNP",
        "amazon_photo-gcn-seed-relative0.1-epochs200": "Amazon GCN",
        "amazon_photo-gat-arch_gat-relative0.1-epochs200": "Amazon GAT",
        "amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200": "Amazon SAGE",
    }
    gnn = gnn[gnn["set_id"].isin(order)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), sharey=False)

    # Panel A: t-statistic comparison (scale-free; degree and diameter share axis)
    ax = axes[0]
    sets = [s for s in order if s in set(gnn["set_id"])]
    y = np.arange(len(sets))
    for term, color, marker, offset in [
        ("rashomon_capacity", PALETTE["diameter"], "o", -0.15),
        ("degree", PALETTE["degree"], "s", 0.15),
    ]:
        xs, ys = [], []
        for i, set_id in enumerate(sets):
            row = gnn[(gnn["set_id"] == set_id) & (gnn["term"] == term)]
            if row.empty:
                continue
            r = row.iloc[0]
            t_stat = r["coefficient"] / r["std_error"] if r["std_error"] else float("nan")
            xs.append(t_stat)
            ys.append(i + offset)
        ax.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            linestyle="none",
            markersize=6,
            label="Probability diameter" if term == "rashomon_capacity" else "Degree",
        )
    ax.axvline(0.0, color="black", linewidth=0.9)
    # Approximate two-sided |t|≈2.1 reference for df≈17
    ax.axvline(2.1, color="#888888", linewidth=0.8, linestyle=":")
    ax.axvline(-2.1, color="#888888", linewidth=0.8, linestyle=":")
    ax.set_yticks(y, [labels[s] for s in sets])
    ax.set_xlabel(r"$t$-statistic on top-$k$ Jaccard (dashed: $|t|\approx 2.1$)")
    ax.set_title("A. GNNExplainer controlled effects")

    # Panel B: Cora GCN node-level scatter
    ax = axes[1]
    nodes = pd.read_csv(
        root
        / "outputs/explanations/cora-gcn-seed-relative0.1-epochs200.gnnexplainer.node_summary.csv"
    )
    sc = ax.scatter(
        nodes["rashomon_capacity"],
        nodes["top_k_jaccard_mean"],
        s=28,
        c=np.log1p(nodes["degree"]),
        cmap="viridis",
        edgecolor="white",
        linewidth=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log(1+$degree$)$")
    ax.set_xlabel("Probability diameter")
    ax.set_ylabel(r"Mean top-$k$ Jaccard")
    ax.set_title("B. Cora GCN nodes")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.38, -0.02),
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    save_fig(fig, output_dir, "fig4_explanation_stability", dpi, extra_artists=[legend])


def _pareto_front(points: pd.DataFrame, acc_col: str, fair_col: str) -> pd.DataFrame:
    """Lower fairness gap is better; higher accuracy is better."""
    ordered = points.sort_values([fair_col, acc_col], ascending=[True, False]).reset_index(drop=True)
    best_acc = -np.inf
    keep = []
    for idx, row in ordered.iterrows():
        if row[acc_col] > best_acc:
            keep.append(idx)
            best_acc = float(row[acc_col])
    return ordered.loc[keep].sort_values(acc_col)


def fig5_nba_fairness_pareto(root: Path, output_dir: Path, dpi: int) -> None:
    """NBA GCN seed Pareto with FairGNN and FairSIN baselines."""
    models = pd.read_csv(root / "outputs/metrics/nba-gcn-seed-relative0.1-epochs200.fairness_models.csv")
    fairgnn = pd.read_csv(root / "outputs/baselines/fairgnn/nba_fairgnn_baseline_models.csv")
    fairsin = pd.read_csv(root / "outputs/baselines/fairsin/nba_fairsin_baseline_models.csv")
    budget = pd.concat(
        [
            pd.read_csv(root / "outputs/budget_matched_fairness/nba_gcn_vs_fairgnn_k5.summary.csv"),
            pd.read_csv(root / "outputs/budget_matched_fairness/nba_gcn_vs_fairsin_k5.summary.csv"),
        ],
        ignore_index=True,
    )
    budget.to_csv(output_dir / "fig5_budget_matched_summary.csv", index=False)

    metric = "validation_demographic_parity_gap"
    front = _pareto_front(models, "validation_accuracy", metric)
    models.to_csv(output_dir / "fig5_nba_gcn_seed_models.csv", index=False)
    front.to_csv(output_dir / "fig5_nba_gcn_seed_pareto.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

    ax = axes[0]
    ax.scatter(
        models[metric],
        models["validation_accuracy"],
        s=28,
        color=PALETTE["neutral"],
        label="Retained GCN",
        zorder=1,
    )
    ax.plot(
        front[metric],
        front["validation_accuracy"],
        "o-",
        color=PALETTE["rashomon"],
        markersize=5,
        linewidth=1.4,
        label="Pareto front",
        zorder=3,
    )
    ax.scatter(
        fairgnn[metric],
        fairgnn["validation_accuracy"],
        s=55,
        marker="D",
        color=PALETTE["fairgnn"],
        label="FairGNN",
        zorder=4,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.scatter(
        fairsin[metric],
        fairsin["validation_accuracy"],
        s=55,
        marker="^",
        color=PALETTE["fairsin"],
        label="FairSIN",
        zorder=4,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Validation demographic-parity gap")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("A. NBA GCN seed frontier")
    ax.legend(frameon=False, loc="lower right")

    # Panel B: budget-matched accuracy win rates
    ax = axes[1]
    seed_rows = budget[budget["set_id"].str.contains("nba-gcn-seed")].copy()
    metrics = ["demographic_parity_gap", "equalized_odds_gap", "degree_disparity"]
    metric_labels = {
        "demographic_parity_gap": "Dem. parity",
        "equalized_odds_gap": "Eq. odds",
        "degree_disparity": "Degree disp.",
    }
    x = np.arange(len(metrics))
    width = 0.35
    for i, (baseline, label) in enumerate([("fairgnn", "vs FairGNN"), ("fairsin", "vs FairSIN")]):
        sub = seed_rows[seed_rows["baseline"] == baseline].set_index("fairness_metric")
        vals = [float(sub.loc[m, "selected_accuracy_win_rate_mean"]) for m in metrics]
        lo = [
            float(
                sub.loc[m, "selected_accuracy_win_rate_mean"]
                - sub.loc[m, "selected_accuracy_win_rate_interval_95_low"]
            )
            for m in metrics
        ]
        hi = [
            float(
                sub.loc[m, "selected_accuracy_win_rate_interval_95_high"]
                - sub.loc[m, "selected_accuracy_win_rate_mean"]
            )
            for m in metrics
        ]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            yerr=[lo, hi],
            color=PALETTE[baseline],
            label=label,
            capsize=2,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    ax.set_xticks(x, [metric_labels[m] for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy win rate (K=5 pool)")
    ax.set_title("B. Budget-matched seed selection")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig5_nba_fairness_pareto", dpi)


def fig6_validation_sensitivity(root: Path, output_dir: Path, dpi: int) -> None:
    """Show how disagreement changes under validation-accuracy constraints."""
    sens = pd.read_csv(
        root / "outputs/performance_sensitivity/performance_constrained_rashomon_summary.csv"
    )
    # Reconstruct base disagreement from parent sets using retained rows at delta=0.05
    # when threshold retains all, else use manuscript-aligned selected cells.
    selected = [
        ("cora", "gcn", "cora-gcn-seed-relative0.1-epochs200"),
        ("citeseer", "gcn", "citeseer-gcn-seed-relative0.1-epochs200"),
        ("pubmed", "gcn", "pubmed-gcn-seed-relative0.1-epochs200"),
        ("pubmed", "gat", "pubmed-gat-canonical_seed-relative2-epochs200"),
        ("amazon_photo", "gcn", "amazon_photo-gcn-hyperparameter-relative2-epochs200"),
        ("nba", "gcn", "nba-gcn-seed-relative0.1-epochs200"),
        ("nba", "gcn", "nba-gcn-hyperparameter-relative2-epochs200"),
        ("nba", "gat", "nba-gat-seed-relative0.1-epochs300"),
    ]
    # Load base multiplicity from rashomon multiplicity summaries where possible
    base_rows = []
    for dataset, architecture, parent in selected:
        parent_path_candidates = [
            root / f"outputs/metrics/{parent}.multiplicity_summary.json",
            root / f"outputs/rashomon_sets/{parent}.json",
        ]
        base_disagreement = None
        base_retained = None
        for candidate in parent_path_candidates:
            if not candidate.exists():
                continue
            payload = json.loads(candidate.read_text())
            if "fraction_prediction_disagreement" in payload:
                base_disagreement = float(payload["fraction_prediction_disagreement"])
                base_retained = int(payload.get("retained_count") or payload.get("n_models") or 0)
                break
            if "summary" in payload and "fraction_prediction_disagreement" in payload["summary"]:
                base_disagreement = float(payload["summary"]["fraction_prediction_disagreement"])
                base_retained = int(payload["summary"].get("retained_count", 0))
                break
            if "metrics" in payload and "fraction_prediction_disagreement" in payload["metrics"]:
                base_disagreement = float(payload["metrics"]["fraction_prediction_disagreement"])
                break
        # Fallback: use largest-delta row's parent metadata via sens table
        sub = sens[sens["parent_set_id"] == parent]
        if sub.empty and parent.endswith("-epochs300"):
            # try epochs200 fallback naming not needed
            pass
        if base_disagreement is None and not sub.empty:
            # Prefer reading from a companion multiplicity if available via train_eligible
            base_retained = int(sub.iloc[0]["train_eligible_count"])
        row_02 = sub[sub["validation_accuracy_delta"] == 0.02]
        row_05 = sub[sub["validation_accuracy_delta"] == 0.05]
        if row_02.empty:
            continue
        # For base disagreement, use the unconstrained value from manuscript tables when JSON missing.
        manuscript_base = {
            "cora-gcn-seed-relative0.1-epochs200": (21, 0.154),
            "citeseer-gcn-seed-relative0.1-epochs200": (13, 0.179),
            "pubmed-gcn-seed-relative0.1-epochs200": (20, 0.145),
            "pubmed-gat-canonical_seed-relative2-epochs200": (50, 0.162),
            "amazon_photo-gcn-hyperparameter-relative2-epochs200": (20, 0.528),
            "nba-gcn-seed-relative0.1-epochs200": (37, 0.293),
            "nba-gcn-hyperparameter-relative2-epochs200": (36, 0.476),
            "nba-gat-seed-relative0.1-epochs300": (20, 0.526),
        }
        retained_base, disag_base = manuscript_base[parent]
        if base_disagreement is not None:
            disag_base = base_disagreement
            if base_retained:
                retained_base = base_retained
        base_rows.append(
            {
                "label": f"{dataset}:{architecture}",
                "parent_set_id": parent,
                "dataset": dataset,
                "architecture": architecture,
                "retained_base": retained_base,
                "disagreement_base": disag_base,
                "retained_0p02": int(row_02.iloc[0]["retained_count"]),
                "disagreement_0p02": float(row_02.iloc[0]["fraction_prediction_disagreement"]),
                "retained_0p05": int(row_05.iloc[0]["retained_count"]) if not row_05.empty else np.nan,
                "disagreement_0p05": float(row_05.iloc[0]["fraction_prediction_disagreement"])
                if not row_05.empty
                else np.nan,
            }
        )
    frame = pd.DataFrame(base_rows)
    frame.to_csv(output_dir / "fig6_validation_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    y = np.arange(len(frame))
    height = 0.28
    bars = [
        ax.barh(
            y + height,
            frame["disagreement_base"],
            height=height,
            color=PALETTE["retained"],
            edgecolor="white",
            linewidth=0.3,
            label="Train-loss set",
        ),
        ax.barh(
            y,
            frame["disagreement_0p05"],
            height=height,
            color=PALETTE["val05"],
            edgecolor="white",
            linewidth=0.3,
            label="Within 0.05 val-acc",
        ),
        ax.barh(
            y - height,
            frame["disagreement_0p02"],
            height=height,
            color=PALETTE["val02"],
            edgecolor="white",
            linewidth=0.3,
            label="Within 0.02 val-acc",
        ),
    ]
    labels = []
    pretty_map = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "amazon_photo": "Amazon-Photo",
        "nba": "NBA",
    }
    for i, row in enumerate(frame.itertuples(index=False)):
        kind = "seed" if "seed" in row.parent_set_id else "hyper"
        if "canonical" in row.parent_set_id:
            kind = "canonical"
        labels.append(f"{pretty_map[row.dataset]} {row.architecture.upper()} ({kind})")
        ax.text(
            max(row.disagreement_0p02, 0) + 0.01,
            i - height,
            f"n_0.02={row.retained_0p02}",
            va="center",
            fontsize=7,
            color="#000000",
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Node disagreement fraction")
    ax.set_title("Validation-accuracy sensitivity")
    ax.set_xlim(0, max(0.55, float(frame[["disagreement_base", "disagreement_0p02", "disagreement_0p05"]].max().max()) + 0.08))
    legend = ax.legend(
        handles=bars,
        labels=["Train-loss set", "Within 0.05 val-acc", "Within 0.02 val-acc"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    save_fig(fig, output_dir, "fig6_validation_sensitivity", dpi, extra_artists=[legend])


def fig8_validation_matched_primary(root: Path, output_dir: Path, dpi: int) -> None:
    """Train-loss vs joint validation-matched disagreement (primary RQ1 figure)."""
    csv_path = output_dir / "fig8_validation_matched_primary.csv"
    if not csv_path.exists():
        csv_path = root / "outputs/paper_figures/fig8_validation_matched_primary.csv"
    frame = pd.read_csv(csv_path)
    order = [
        "cora-gcn-seed-relative0.1-epochs200",
        "citeseer-gcn-seed-relative0.1-epochs200",
        "pubmed-gcn-seed-relative0.1-epochs200",
        "amazon_photo-gcn-hyperparameter-relative2-epochs200",
        "nba-gcn-hyperparameter-relative2-epochs200",
        "ogbn_arxiv-gcn-seed-relative0.1-epochs300",
    ]
    labels = {
        "cora-gcn-seed-relative0.1-epochs200": "Cora seed",
        "citeseer-gcn-seed-relative0.1-epochs200": "CiteSeer seed",
        "pubmed-gcn-seed-relative0.1-epochs200": "PubMed seed",
        "amazon_photo-gcn-hyperparameter-relative2-epochs200": "Amazon-Photo HP",
        "nba-gcn-hyperparameter-relative2-epochs200": "NBA HP",
        "ogbn_arxiv-gcn-seed-relative0.1-epochs300": "ogbn-arxiv seed",
    }
    present = [s for s in order if s in set(frame["parent_set_id"])]
    y = np.arange(len(present))
    height = 0.35
    train = []
    joint = []
    for set_id in present:
        sub = frame[frame["parent_set_id"] == set_id]
        train.append(float(sub.loc[sub["definition"] == "train_loss_only", "disagreement"].iloc[0]))
        joint.append(float(sub.loc[sub["definition"] == "joint_delta0.02", "disagreement"].iloc[0]))
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    bars = [
        ax.barh(
            y + height / 2,
            joint,
            height=height,
            color=PALETTE["joint"],
            edgecolor="white",
            linewidth=0.3,
            label=r"Joint $\mathcal{R}_{0.02}$",
        ),
        ax.barh(
            y - height / 2,
            train,
            height=height,
            color=PALETTE["train_loss"],
            edgecolor="white",
            linewidth=0.3,
            label="Train-loss only",
        ),
    ]
    ax.set_yticks(y, [labels[s] for s in present])
    ax.set_xlabel("Class disagreement")
    ax.set_title("Validation-matched primary reporting")
    legend = ax.legend(handles=bars, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    frame.to_csv(output_dir / "fig8_validation_matched_primary.csv", index=False)
    save_fig(fig, output_dir, "fig8_validation_matched_primary", dpi, extra_artists=[legend])


def fig9_node_delta_tails(root: Path, output_dir: Path, dpi: int) -> None:
    """Node-level practical thresholds under random rewiring strength 0.25."""
    csv_path = output_dir / "fig9_node_delta_tails.csv"
    if not csv_path.exists():
        csv_path = root / "outputs/paper_figures/fig9_node_delta_tails.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        src = pd.read_csv(root / "outputs/rewiring_effect_sizes/rewiring_effect_sizes_node_delta_summary.csv")
        src = src[src["stem"].str.contains(r"-random-0p25-seed", regex=True)].copy()
        src["dataset"] = src["stem"].str.split("-").str[0]
        frame = (
            src.groupby("dataset", as_index=False)[
                [
                    "fraction_delta_D_i_positive",
                    "fraction_delta_D_i_gt_0p01",
                    "fraction_delta_D_i_gt_0p05",
                    "fraction_delta_D_i_gt_0p1",
                ]
            ]
            .mean()
        )
    frame = frame.set_index("dataset").reindex(DATASET_ORDER).reset_index()
    metrics = [
        ("fraction_delta_D_i_positive", r"$\Delta D_i>0$"),
        ("fraction_delta_D_i_gt_0p01", r"$>0.01$"),
        ("fraction_delta_D_i_gt_0p05", r"$>0.05$"),
        ("fraction_delta_D_i_gt_0p1", r"$>0.10$"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, dataset in enumerate(DATASET_ORDER):
        row = frame[frame["dataset"] == dataset].iloc[0]
        vals = [100.0 * float(row[col]) for col, _ in metrics]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=PALETTE[dataset],
            edgecolor="white",
            linewidth=0.3,
            label=DATASET_LABELS[dataset],
        )
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Share of nodes (%)")
    ax.set_title(r"Node-level $\Delta D_i$ tails under random rewiring (strength 0.25)")
    ax.set_ylim(0, 80)
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    frame.to_csv(output_dir / "fig9_node_delta_tails.csv", index=False)
    save_fig(fig, output_dir, "fig9_node_delta_tails", dpi, extra_artists=[legend])


def write_manifest(output_dir: Path, paths: list[Path]) -> None:
    lines = [
        "# Paper figures",
        "",
        "Generated by `scripts/paper_figures/generate_paper_figures.py`.",
        "",
        "| Figure | File | Role in manuscript |",
        "|---|---|---|",
        "| Fig 1 | `fig1_structural_correlates` | RQ2 observational correlates |",
        "| Fig 2 | `fig2_rewiring_dose_response` | RQ2 random rewiring dose response (w/ validity shading) |",
        "| Fig 3 | `fig3_local_matched_effects` | RQ2 local matched intervention |",
        "| Fig 4 | `fig4_explanation_stability` | RQ4 multiplicity vs degree for XAI |",
        "| Fig 5 | `fig5_nba_fairness_pareto` | RQ5 NBA selection vs FairGNN/FairSIN |",
        "| Fig 6 | `fig6_validation_sensitivity` | RQ3 tolerance / val-acc sensitivity |",
        "| Fig 7 | `fig7_relative_dose_response` | RQ2 relative/absolute dose + invalid region |",
        "| Fig 8 | `fig8_validation_matched_primary` | RQ1 train-loss vs joint $\\delta=0.02$ |",
        "| Fig 9 | `fig9_node_delta_tails` | RQ2 node-level practical thresholds |",
        "",
        "## Generated files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    written: list[Path] = []
    fig1_structural_correlates(root, output_dir, args.dpi)
    fig2_rewiring_dose_response(root, output_dir, args.dpi)
    fig3_local_matched_effects(root, output_dir, args.dpi)
    fig4_explanation_stability(root, output_dir, args.dpi)
    fig5_nba_fairness_pareto(root, output_dir, args.dpi)
    fig6_validation_sensitivity(root, output_dir, args.dpi)
    fig7_relative_dose_response(root, output_dir, args.dpi)
    fig8_validation_matched_primary(root, output_dir, args.dpi)
    fig9_node_delta_tails(root, output_dir, args.dpi)
    written = sorted(output_dir.glob("fig*.*"))
    write_manifest(output_dir, written)
    print(f"Done. Figures in {output_dir}")


if __name__ == "__main__":
    main()
