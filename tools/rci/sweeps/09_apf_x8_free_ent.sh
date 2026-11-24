#!/bin/bash
set -euo pipefail

SEEDS="0 1"

for ent in 0.001 0; do
    python3 tools/rci/submit.py --walltime 1:00:00 --seeds $SEEDS -- \
        --task chase --preset apf_x8_free_3v1 --steps 5e8 \
        --set normalize_obs=false --set adaptive_lr=false \
        --set lr=1.25e-4 --set ent_coef=$ent \
        --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
        --out runs/09_apf_x8_free_ent/ent_coef$ent
done
