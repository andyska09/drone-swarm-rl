#!/bin/bash -l
# SLURM job for one run/train.py call. Submit with submit.py, which sets every
# resource on the sbatch line and passes the train.py arguments to this script.
# Results go straight into runs/ in the repo on /mnt/personal.

# The python of the venv needs this module at run time. ml exists only in a login
# shell, so line 1 has -l.
ml Python/3.12.3-GCCcore-13.3.0
set -euo pipefail

PYTHON=/mnt/personal/$USER/drone-swarm-env/bin/python

echo "=== $SLURM_JOB_ID  $(hostname -f)  $(date) ==="
echo "ARGS: $*"
nvidia-smi -L
# A node with a stuck GPU hangs here until the walltime ends.
timeout 300 "$PYTHON" -c "import jax; print(jax.devices())"

"$PYTHON" run/train.py "$@"

# train.py ends with the line "wrote <run folder>", and submit.py sends it to this log.
RUN=$(grep '^wrote ' "logs/$SLURM_JOB_ID.out" | tail -n 1 | cut -d ' ' -f 2)
"$PYTHON" run/eval.py "$RUN" --episodes 4096

echo "=== done $(date) ==="
