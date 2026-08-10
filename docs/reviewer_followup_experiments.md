# Reviewer follow-up experiments

This note packages the highest-priority reviewer requests against the current
`gnn-rashomon` codebase. **No jobs were submitted from this session.**

## Experiment #1 — Counterfactual graph retraining (GPU)

### Design

- Datasets: Cora, CiteSeer, PubMed
- Architecture: GCN only
- Train seeds: `0..49` (same ensemble size / hyperparameters as baseline)
- Graph seeds: `0..4` (5 counterfactual graphs per condition)
- Conditions:
  - `original` — existing baseline sets (no retrain)
  - `random_0p05` — mild random rewiring
  - `random_0p25` — stronger random rewiring
  - `local_tau0p1` — local entropy-targeted rewiring
- Graphs reused from the fixed-ensemble repeated-rewiring tree (canonical
  Planetoid `NormalizeFeatures` path), avoiding the withdrawn preprocessing bug
- Training artifacts isolated under `outputs/counterfactual_retraining/`
- Metrics compared for \(R(G')\) vs \(R(G)\):
  retained-set size, class disagreement, mean probability diameter, mean pairwise
  TV, validation-accuracy distribution

Job count: **2250** GPU train tasks (`3 × 3 × 5 × 50`).

### Launch (separate clusters; no cross-cluster dependency)

On **alpha** — one GPU job runs all 2250 trainings internally:

```bash
cd ${PROJECT_DIR}
bash scripts/slurm/submit_cf_retrain_alpha.sh
# equivalent:
# sbatch scripts/slurm/cf_retrain_seed_alpha_gpu.sbatch
```

The job requests 7 days / 1 GPU and skips tasks that already have metrics, so
requeues or partial resubmits resume cleanly. Optional range:

```bash
CF_RETRAIN_START_TASK=0 CF_RETRAIN_END_TASK=749 sbatch scripts/slurm/cf_retrain_seed_alpha_gpu.sbatch
```

(The old 2250-task array script remains as
`cf_retrain_seed_alpha_gpu_array.sbatch` if needed.)

After training finishes, on **Romeo** (analysis only):

```bash
cd ${PROJECT_DIR}
bash scripts/slurm/submit_cf_retrain_analyze_romeo.sh
# equivalent:
# sbatch scripts/slurm/cf_retrain_analyze_romeo_cpu.sbatch
```

Smoke one decoded task locally before the full array:

```bash
export SLURM_ARRAY_TASK_ID=0
# inspect decode
python - <<'PY'
from gnn_rashomon.analysis.counterfactual_retraining import decode_array_task, graph_path_for, rashomon_type_for
d,c,g,s = decode_array_task(0)
print(d, c.condition_id, g, s, graph_path_for(d,c,g), rashomon_type_for(c,g))
PY
```

### After training

Analysis job builds sets + multiplicity and writes:

- `outputs/counterfactual_retraining/cf_retrain_summary_detail.csv`
- `outputs/counterfactual_retraining/cf_retrain_summary_vs_baseline.csv`
- `outputs/counterfactual_retraining/cf_retrain_summary_by_condition.csv`

Manual summarize (if analysis job already built the sets):

```bash
python -m gnn_rashomon.cli.summarize_counterfactual_retraining
```

---

## Reanalysis package (#2, #4, #5, #6) — no new training

Run these on the Romeo/module environment (CPU). Commands only; do not submit from
this workspace unless you intend to.

### Shared environment

```bash
cd ${PROJECT_DIR}
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
source .venv-module/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

### One-shot Slurm job (#2+#4+#5+#6)

On Romeo:

```bash
sbatch scripts/slurm/reviewer_reanalysis_romeo_cpu.sbatch
```

This writes:

- `outputs/rashomon_membership_sensitivity/`
- `outputs/rewiring_effect_sizes/`
- `outputs/equivalence/`

Optional node-level \(\Delta D_i\) for #4 remains a separate array
(`materialize_rewiring_node_deltas_romeo_cpu_array.sbatch`); if those CSVs
already exist, the one-shot job will include them automatically.

### #2 + #6 — Validation-matched primary tables + definition ablation

Joint membership:

\[
\mathcal{R}_\delta=
\{h: L_{\mathrm{train}}(h)\le(1+\epsilon)L^*,\
\mathrm{Acc}_{val}(h)\ge \mathrm{Acc}_{val}^*-\delta\}
\]

with \(\delta\in\{0.01,0.02,0.05\}\), **primary \(\delta=0.02\)**. Also emits
train-loss-only and validation-only ablations.

```bash
python -m gnn_rashomon.cli.build_rashomon_membership_sensitivity \
  --validation-accuracy-delta 0.01 \
  --validation-accuracy-delta 0.02 \
  --validation-accuracy-delta 0.05 \
  --primary-delta 0.02 \
  --output-dir outputs/rashomon_membership_sensitivity
```

Outputs:

- `outputs/rashomon_membership_sensitivity/membership_sensitivity_all.csv`
- `outputs/rashomon_membership_sensitivity/primary_joint_validation_matched.csv`
- `outputs/rashomon_membership_sensitivity/definition_ablation.csv`

### #4 — Rewiring practical effect sizes + dose-response

Graph-level relative effects from existing realization CSV (immediate):

```bash
python -m gnn_rashomon.cli.build_rewiring_effect_sizes \
  --realizations-csv outputs/repeated_rewiring/repeated_rewiring_realizations.csv \
  --output-prefix outputs/rewiring_effect_sizes/rewiring_effect_sizes
```

Outputs:

- `..._realizations_enriched.csv` — absolute + relative \(\Delta D\)
- `..._summary.csv` — by dataset/mode/strength
- `..._dose_response.csv` — random-rewiring strength curve (0.01/0.05/0.10/0.25)

Optional node-level \(\Delta D_i\) quantiles (requires rematerializing node tables):

```bash
sbatch scripts/slurm/materialize_rewiring_node_deltas_romeo_cpu_array.sbatch
# then
python -m gnn_rashomon.cli.build_rewiring_effect_sizes \
  --realizations-csv outputs/repeated_rewiring/repeated_rewiring_realizations.csv \
  --node-delta-glob 'outputs/repeated_rewiring/random/node_deltas/*.csv' \
  --output-prefix outputs/rewiring_effect_sizes/rewiring_effect_sizes
```

### #5 — RQ3 equivalence / TOST

Default SESOI: \(|\beta_{\mathrm{rashomon\_capacity}}| < 0.10\) on mean top-\(k\) Jaccard.

```bash
python -m gnn_rashomon.cli.run_equivalence_tests \
  --equivalence-margin 0.10 \
  --output-prefix outputs/equivalence/rq3_equivalence
```

Outputs:

- `outputs/equivalence/rq3_equivalence.csv`
- `outputs/equivalence/rq3_equivalence.summary.json`

---

## Suggested order on Romeo (CPU)

1. `#2+#6` membership sensitivity
2. `#4` graph-level effect sizes
3. `#5` equivalence tests
4. (optional) materialize node deltas, then rerun `#4` with `--node-delta-glob`

## Suggested order on alpha (GPU)

1. Submit `#1` training on alpha (`submit_cf_retrain_alpha.sh`), then analysis on Romeo (`submit_cf_retrain_analyze_romeo.sh`)
2. Inspect `cf_retrain_summary_by_condition.csv`
3. Only then consider PubMed-GAT replication (#7)
