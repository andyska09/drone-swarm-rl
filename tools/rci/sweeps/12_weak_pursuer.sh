#!/bin/bash
set -euo pipefail

SEEDS="0 1"

for preset in apf_1v1_weak apf_3v1_weak; do
    python3 tools/rci/submit.py --walltime 1:00:00 --seeds $SEEDS -- \
        --task chase --preset $preset --steps 5e8 \
        --set normalize_obs=false --set adaptive_lr=false \
        --set lr=1.25e-4 --set ent_coef=0.001 \
        --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
        --out runs/12_weak_pursuer/$preset
done
