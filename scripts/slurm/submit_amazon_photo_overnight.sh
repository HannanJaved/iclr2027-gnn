#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p .logs outputs/figures

AMZ_ARCH_SMOKE=$(sbatch --parsable scripts/slurm/smoke_amazon_photo_arch_module_cpu.sbatch)
AMZ_ARCH_SEEDS=$(sbatch --parsable \
  --dependency=afterok:${AMZ_ARCH_SMOKE} \
  scripts/slurm/seed_amazon_photo_arch_module_cpu_array.sbatch)
AMZ_ARCH_ANALYSIS=$(sbatch --parsable \
  --dependency=afterok:${AMZ_ARCH_SEEDS} \
  scripts/slurm/analysis_amazon_photo_arch_module_cpu.sbatch)

AMZ_GCN_SMOKE=$(sbatch --parsable scripts/slurm/smoke_amazon_photo_module_cpu.sbatch)
AMZ_GCN_SEEDS=$(sbatch --parsable \
  --dependency=afterok:${AMZ_GCN_SMOKE} \
  scripts/slurm/seed_amazon_photo_module_cpu_array.sbatch)
AMZ_GCN_ANALYSIS=$(sbatch --parsable \
  --dependency=afterok:${AMZ_GCN_SEEDS} \
  scripts/slurm/analysis_amazon_photo_gcn.sbatch)
AMZ_GCN_HP=$(sbatch --parsable \
  --dependency=afterok:${AMZ_GCN_SMOKE} \
  scripts/slurm/hyperparameter_amazon_photo_module_cpu_array.sbatch)
AMZ_GCN_HP_ANALYSIS=$(sbatch --parsable \
  --dependency=afterok:${AMZ_GCN_HP} \
  scripts/slurm/analysis_amazon_photo_gcn_hyperparameter.sbatch)

AMZ_ARCH_TOLERANCE=$(sbatch --parsable \
  --dependency=afterok:${AMZ_ARCH_ANALYSIS}:${AMZ_GCN_ANALYSIS}:${AMZ_GCN_HP_ANALYSIS} \
  scripts/slurm/analysis_amazon_photo_arch_tolerance.sbatch)
AMZ_APPNP_DIAG=$(sbatch --parsable \
  --dependency=afterok:${AMZ_ARCH_ANALYSIS} \
  scripts/slurm/diagnose_amazon_photo_appnp_degeneracy.sbatch)

AMZ_GCN_XAI=$(sbatch --parsable \
  --dependency=afterok:${AMZ_GCN_ANALYSIS} \
  scripts/slurm/explanation_amazon_photo_gcn.sbatch)
AMZ_ARCH_XAI=$(sbatch --parsable \
  --dependency=afterok:${AMZ_ARCH_ANALYSIS} \
  scripts/slurm/explanation_amazon_photo_arch.sbatch)

STAT_AUDIT=$(sbatch --parsable \
  --dependency=afterok:${AMZ_ARCH_ANALYSIS}:${AMZ_GCN_ANALYSIS}:${AMZ_GCN_HP_ANALYSIS}:${AMZ_ARCH_TOLERANCE}:${AMZ_APPNP_DIAG}:${AMZ_GCN_XAI}:${AMZ_ARCH_XAI} \
  scripts/slurm/statistical_audit_stage1.sbatch)
MANIFEST=$(sbatch --parsable \
  --dependency=afterok:${STAT_AUDIT} \
  scripts/slurm/build_repro_manifest_stage1.sbatch)

cat <<EOF
Amazon-Photo overnight queue submitted.

Architecture smoke:       ${AMZ_ARCH_SMOKE}
Architecture seeds:       ${AMZ_ARCH_SEEDS}
Architecture analysis:    ${AMZ_ARCH_ANALYSIS}
GCN smoke:                ${AMZ_GCN_SMOKE}
GCN seeds:                ${AMZ_GCN_SEEDS}
GCN analysis:             ${AMZ_GCN_ANALYSIS}
GCN hyperparameter:       ${AMZ_GCN_HP}
GCN HP analysis:          ${AMZ_GCN_HP_ANALYSIS}
Architecture tolerance:   ${AMZ_ARCH_TOLERANCE}
APPNP diagnostic:         ${AMZ_APPNP_DIAG}
GCN explanations:         ${AMZ_GCN_XAI}
Architecture explanations:${AMZ_ARCH_XAI}
Statistical audit:        ${STAT_AUDIT}
Manifest:                 ${MANIFEST}
EOF
