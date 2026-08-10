#!/usr/bin/env bash
set -euo pipefail

python -m gnn_rashomon.cli.train_baseline dataset=cora trainer.max_epochs=5
