#!/bin/bash
set -euo pipefail

LRS="1.25e-4 2.5e-4 5e-4 1e-3"
ENTS="0.001 0.003 0.01 0.03"
SEEDS="0 1"

for lr in $LRS; do
    for ent in $ENTS; do
        python3 tools/rci/submit.py --walltime 1:00:00 --seeds $SEEDS -- \
            --task chase --preset paths_3v1 --steps 5e8 \
            --set gamma=0.99 --set normalize_obs=false \
            --set adaptive_lr=false --set lr=$lr --set ent_coef=$ent \
            --out runs/02_lr_ent/lr${lr}_ent${ent}
    done
done
