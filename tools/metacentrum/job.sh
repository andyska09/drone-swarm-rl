#!/bin/bash
# PBS job for one run/train.py call. Submit with submit.py, which sets every
# resource on the qsub line plus ROOT (the repo) and ARGS (the train.py arguments).
# Results go straight into $ROOT/runs on shared home.
set -euo pipefail

cd "$ROOT"
conda activate "$PBS_O_HOME/.conda/envs/drone-swarm"

echo "=== $PBS_JOBID  $(hostname -f)  $(date) ==="
echo "ARGS: $ARGS"
nvidia-smi -L
python -c "import jax; print(jax.devices())"

# Unquoted on purpose: each word is one train.py argument.
python run/train.py $ARGS

echo "=== done $(date) ==="
