#!/usr/bin/env bash
#SBATCH --job-name=cf_retrain_seed
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

# Counterfactual retraining (#1): single alpha GPU job that runs all 2250
# trainings in-process (3 datasets × 3 conditions × 5 graph seeds × 50 seeds).
# Skips tasks that already have a matching metrics record so requeues/resubmits
# can resume. Failures are logged and skipped so one bad run does not abort the job.

set -euo pipefail
export PYTHONUNBUFFERED=1

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

OUTPUT_DIR="${PROJECT_DIR}/outputs/counterfactual_retraining"
METRICS_DIR="${OUTPUT_DIR}/metrics"
mkdir -p "${METRICS_DIR}" .logs

export PROJECT_DIR
export OUTPUT_DIR
export START_TASK="${CF_RETRAIN_START_TASK:-0}"
export END_TASK="${CF_RETRAIN_END_TASK:-2249}"
TOTAL_TASKS=$((END_TASK - START_TASK + 1))

echo "[$(date -Is)] starting cf-retrain monolithic job tasks ${START_TASK}-${END_TASK} (${TOTAL_TASKS} tasks)"

python -u - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gnn_rashomon.analysis.counterfactual_retraining import (
    decode_array_task,
    expected_job_count,
    graph_path_for,
    rashomon_type_for,
)
from gnn_rashomon.rashomon.io import discover_runs

project_dir = Path(os.environ["PROJECT_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
metrics_dir = output_dir / "metrics"
start_task = int(os.environ["START_TASK"])
end_task = int(os.environ["END_TASK"])
assert expected_job_count() == 2250

done = 0
skipped = 0
failed = 0
ran = 0

for task_id in range(start_task, end_task + 1):
    dataset, condition, graph_seed, train_seed = decode_array_task(task_id)
    graph_path = project_dir / graph_path_for(dataset, condition, graph_seed)
    rashomon_type = rashomon_type_for(condition, graph_seed)
    existing = discover_runs(
        metrics_dir=metrics_dir,
        dataset=dataset,
        architecture="gcn",
        max_epochs=200,
        rashomon_type=rashomon_type,
    )
    if any(int(run.get("seed", -1)) == train_seed for run in existing):
        skipped += 1
        done += 1
        if task_id == start_task or task_id == end_task or task_id % 50 == 0:
            print(
                f"[task {task_id}/{end_task}] skip existing "
                f"{dataset} {rashomon_type} seed={train_seed}",
                flush=True,
            )
        continue

    if not graph_path.is_file():
        failed += 1
        done += 1
        print(f"[task {task_id}/{end_task}] FAIL missing graph {graph_path}", flush=True)
        continue

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "gnn_rashomon.cli.train_baseline",
        f"dataset={dataset}",
        f"graph_path={graph_path}",
        f"rashomon_type={rashomon_type}",
        f"seed={train_seed}",
        f"paths.output_dir={output_dir}",
        "trainer.device=cuda",
        "trainer.max_epochs=200",
    ]
    print(
        f"[task {task_id}/{end_task}] train {dataset} {condition.condition_id} "
        f"gseed={graph_seed} seed={train_seed}",
        flush=True,
    )
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(project_dir))
    elapsed = time.time() - t0
    if result.returncode != 0:
        failed += 1
        print(
            f"[task {task_id}/{end_task}] FAIL rc={result.returncode} after {elapsed:.1f}s",
            flush=True,
        )
    else:
        ran += 1
        print(f"[task {task_id}/{end_task}] ok in {elapsed:.1f}s", flush=True)
    done += 1

summary = {
    "start_task": start_task,
    "end_task": end_task,
    "done": done,
    "ran": ran,
    "skipped_existing": skipped,
    "failed": failed,
}
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
(output_dir / "cf_retrain_monolithic_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)
if failed:
    raise SystemExit(f"{failed} training tasks failed")
PY
