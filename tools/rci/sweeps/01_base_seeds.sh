#!/bin/bash
# Sweep 01_base_seeds: 8 seeds of the base config, to measure how much the catch moves
# from the seed alone. Run it on login3 from the repo root.
set -euo pipefail

python3 tools/rci/submit.py --walltime 1:00:00 --seeds 0 1 2 3 4 5 6 7 -- \
    --task chase --preset paths_3v1 --steps 5e8 \
    --set gamma=0.99 --set normalize_obs=false \
    --set adaptive_lr=false --set lr=5e-4 --set ent_coef=0.003 \
    --out runs/01_base_seeds/base
