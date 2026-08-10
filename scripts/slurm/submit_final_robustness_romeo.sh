#!/usr/bin/env bash
set -euo pipefail

set_size_job=$(sbatch --parsable scripts/slurm/set_size_robustness_romeo_cpu_array.sbatch)
set_size_summary_job=$(
  sbatch --parsable --dependency="afterok:${set_size_job}" \
    scripts/slurm/summarize_set_size_robustness_romeo_cpu.sbatch
)
fairness_job=$(sbatch --parsable scripts/slurm/budget_matched_nba_fairness_romeo_cpu.sbatch)
gcn_summary_job=$(sbatch --parsable scripts/slurm/summarize_repeated_rewiring_romeo_cpu.sbatch)
gat_training_job=$(
  sbatch --parsable scripts/slurm/pubmed_gat_canonical_seed_romeo_cpu_array.sbatch
)
prepare_job=$(
  sbatch --parsable --dependency="afterok:${gat_training_job}" \
    scripts/slurm/pubmed_gat_rewiring_prepare_romeo_cpu.sbatch
)
rewiring_job=$(
  sbatch --parsable --dependency="afterok:${prepare_job}" \
    scripts/slurm/pubmed_gat_fixed_rewiring_romeo_cpu_array.sbatch
)
summary_job=$(
  sbatch --parsable --dependency="afterok:${rewiring_job}" \
    scripts/slurm/summarize_pubmed_gat_rewiring_romeo_cpu.sbatch
)

printf 'set-size robustness: %s\n' "${set_size_job}"
printf 'set-size summary: %s\n' "${set_size_summary_job}"
printf 'budget-matched fairness: %s\n' "${fairness_job}"
printf 'GCN rewiring nonparametric summary: %s\n' "${gcn_summary_job}"
printf 'canonical PubMed GAT training: %s\n' "${gat_training_job}"
printf 'PubMed GAT preparation: %s\n' "${prepare_job}"
printf 'PubMed GAT rewiring array: %s\n' "${rewiring_job}"
printf 'PubMed GAT summary: %s\n' "${summary_job}"
