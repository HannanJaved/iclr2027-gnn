# SLURM Jobs

Submit from anywhere with:

```bash
sbatch ${PROJECT_DIR}/scripts/slurm/install_dev.sbatch
sbatch ${PROJECT_DIR}/scripts/slurm/smoke_cora.sbatch
sbatch ${PROJECT_DIR}/scripts/slurm/baseline_cora.sbatch
sbatch ${PROJECT_DIR}/scripts/slurm/seed_cora_array.sbatch
sbatch ${PROJECT_DIR}/scripts/slurm/rashomon_cora_seed.sbatch
sbatch ${PROJECT_DIR}/scripts/slurm/hyperparameter_cora_array.sbatch
sbatch ${PROJECT_DIR}/scripts/slurm/hyperparameter_pubmed_array.sbatch
```

Current implemented experiment scripts target Cora and PubMed GCN training. Pokec,
hyperparameter sweeps, rewiring, explanations, and fairness jobs should be added
once the corresponding CLIs are implemented.

For a larger overnight queue, submit seed and hyperparameter arrays first, then
submit the matching `rashomon_*` post-processing scripts after their arrays finish.
# Reviewer-Risk Extensions

The following jobs add the large-scale and fairness robustness experiments without changing the
completed Stage-1 artifacts. All jobs run on `alpha` and request one GPU per task, including setup
and post-processing jobs required by the partition policy.

```bash
# Submit the complete OGB setup, validation, GPU training, and analysis chain on Alpha.
bash scripts/slurm/submit_reviewer_extensions_alpha.sh

# Independently submit only the NBA GAT and FairSIN chain on Alpha.
bash scripts/slurm/submit_nba_extensions_alpha.sh
```

The launchers are independent and can be submitted from separate SSH connections. The OGB launcher
installs OGB into the isolated project-local `.alpha-deps` directory and owns all `ogbn-arxiv`
stages. The NBA launcher submits only the GAT seed array, GAT hyperparameter array, dependent
fairness analysis, and independent FairSIN job. Dependencies exist only among jobs submitted by
the same launcher. Alpha setup and analysis jobs reserve one GPU even when computation is CPU-bound.
To submit pieces manually instead:

```bash
# Install OGB in Alpha's isolated dependency directory.
sbatch scripts/slurm/install_ogbn_arxiv_alpha.sbatch

# Downloads/validates the official OGB split on Alpha.
sbatch scripts/slurm/validate_ogbn_arxiv.sbatch

# Submit after validation. The array requests one GPU per task on Alpha;
# structural analysis also remains on Alpha.
sbatch scripts/slurm/seed_ogbn_arxiv_array.sbatch
sbatch scripts/slurm/analysis_ogbn_arxiv_stage1.sbatch

# NBA GAT fairness extension.
sbatch scripts/slurm/seed_nba_gat_array.sbatch
sbatch scripts/slurm/hyperparameter_nba_gat_array.sbatch
sbatch scripts/slurm/analysis_nba_gat_fairness.sbatch

# Model-centric FairSIN baseline and comparison with existing NBA GCN Pareto tables.
sbatch scripts/slurm/fairsin_nba_baseline_alpha.sbatch
```

`ogbn-arxiv` requires the `ogb` package now declared in `pyproject.toml`. Its structural job uses
`--profile scalable`, which computes degree, local homophily, neighborhood label entropy,
confidence, and correctness while explicitly omitting clustering, feature similarity, and
community detection.

## Reviewer-Extension Diagnostics

The completed reviewer extensions must be archived and audited before any diagnostic reruns. No
launcher or Slurm dependency is used. Submit the snapshot first and wait for it to complete before
submitting the remaining independent jobs.

```bash
# 1. Hash the completed OGB, NBA GAT, and FairSIN artifacts before adding diagnostics.
sbatch scripts/slurm/snapshot_reviewer_extensions_alpha.sbatch
```

After the snapshot has disappeared from `squeue` and its log reports a written manifest:

```bash
# 2. Audit completed results. These two jobs can run independently.
sbatch scripts/slurm/audit_fairsin_comparison_alpha.sbatch
sbatch scripts/slurm/audit_nba_gat_existing_alpha.sbatch

# 3. Submit the two isolated recipe smoke arrays independently.
sbatch scripts/slurm/smoke_ogbn_arxiv_recipe_array.sbatch
sbatch scripts/slurm/smoke_nba_gat_recipe_array.sbatch
```

After both smoke arrays have completed and disappeared from `squeue`, submit the summary manually:

```bash
sbatch scripts/slurm/summarize_reviewer_smokes_alpha.sbatch
```

After the smoke summaries pass their health checks, submit the isolated full reruns and their final
provenance snapshot with:

```bash
bash scripts/slurm/submit_revised_extensions_alpha.sh
```

This wrapper uses the smoke-selected OGB width-256 recipe and the noncollapsed NBA GAT region. It
does not rerun FairSIN or mix the revised candidates with the earlier diagnostic artifacts.

The OGB smoke uses the official split without row-normalizing the pretrained dense node features
and compares hidden widths 64 and 256 over three seeds. The NBA GAT smoke evaluates four lower-
learning-rate, lower-dropout recipes over three seeds. Diagnostic runs use dedicated
`rashomon_type` values and 300 epochs, preventing them from entering the completed 200-epoch seed
or hyperparameter sets. The health summaries report retained counts, validation utility,
constant-class collapse, disagreement, capacity, and report-only test accuracy across tolerances.

For the canonical-environment audit, first regenerate the focal artifact tree under the canonical
environment in a separate root. Then submit:

```bash
REFERENCE_ROOT=/path/to/archived/focal \
CANDIDATE_ROOT=/path/to/canonical/rerun \
sbatch --export=ALL,REFERENCE_ROOT,CANDIDATE_ROOT \
  scripts/slurm/audit_canonical_environment.sbatch
```

The audit fails the job on any missing or nonequivalent CSV, JSON, or tensor artifact and writes
`outputs/metadata/canonical_environment_equivalence.json`.
