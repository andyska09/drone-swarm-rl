#!/bin/bash
set -euo pipefail

SEEDS="0 1"

run() {
    preset=$1
    python3 tools/rci/submit.py --walltime 1:00:00 --seeds $SEEDS -- \
        --task chase --preset $preset --steps 5e8 \
        --set normalize_obs=false --set adaptive_lr=false \
        --set lr=1.25e-4 --set ent_coef=0.003 \
        --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
        --out runs/08_apf_x8_free/$preset
}

run apf_x8_free_1v1
run apf_x8_free_3v1
