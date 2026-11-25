#!/bin/bash
set -euo pipefail

SEEDS="0 1"

run() {
    label=$1
    shift
    python3 tools/rci/submit.py --walltime 2:00:00 --seeds $SEEDS -- \
        --task chase --preset apf_3v1_weak --steps 2e9 \
        --set normalize_obs=false --set adaptive_lr=false \
        --set lr=1.25e-4 --set ent_coef=0.001 \
        --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
        "$@" \
        --out runs/13_weak_pursuer_long/apf_3v1_weak$label
}

run ""
run _v6_a13 --set-env weak_air_resistance_coeff=3.773
