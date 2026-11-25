#!/bin/bash
set -euo pipefail

SEEDS="0 1"

run() {
    steps=$1
    walltime=$2
    python3 tools/rci/submit.py --walltime $walltime --seeds $SEEDS -- \
        --task chase --preset 3v1 --steps $steps \
        --set generations=13 \
        --set normalize_obs=false --set adaptive_lr=false \
        --set lr=1.25e-4 --set ent_coef=0.001 \
        --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
        --out runs/01_pool_steps/steps$steps
}

run 1.4e9 1:30:00
