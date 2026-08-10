# GNN Rashomon (anonymous code release)

Anonymous supplementary code for the ICLR submission on predictive multiplicity
in graph neural networks: empirical Rashomon sets, structural localization,
fixed-ensemble degree-preserving rewiring, explanation stability checks, and
fairness / deployment analyses.

This release contains source code, configs, tests, documentation, and helper
scripts. Large trained checkpoints, downloaded datasets, and full experiment
outputs are **not** bundled; they can be regenerated with the commands below
(or shared after the review period).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

GPU users: install a matching PyTorch build first, then install this package.
See `requirements-canonical.txt` for a pinned reference stack used in the paper
runs.

## Quick start

```bash
# Minimal Cora smoke training
python -m gnn_rashomon.cli.train_baseline dataset=cora trainer.max_epochs=5

# Build a seed Rashomon set (after completed training jobs)
python -m gnn_rashomon.cli.generate_rashomon \
  --dataset cora --architecture gcn --set-type seed --rashomon-type seed \
  --max-epochs 200 --tolerance-mode relative --epsilon 0.1

# Multiplicity + structure
python -m gnn_rashomon.cli.compute_multiplicity \
  outputs/rashomon_sets/cora-gcn-seed-relative0.1-epochs200.json
```

Paper figure regeneration (once artifacts are present):

```bash
python scripts/paper_figures/generate_paper_figures.py
```

## Layout

- `src/gnn_rashomon/` — library and CLIs
- `configs/` — experiment configs
- `scripts/` — local helper scripts (e.g., paper figures)
- `docs/` — experiment notes used during development
- `tests/` — unit / regression tests
- `outputs/` — empty placeholders for local runs

## License

Released for anonymous peer review. See `LICENSE`.
