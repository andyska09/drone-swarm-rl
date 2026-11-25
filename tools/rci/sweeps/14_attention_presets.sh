#!/bin/bash
set -euo pipefail

SEEDS="0 1"

run() {
    label=$1
    walltime=$2
    shift 2
    python3 tools/rci/submit.py --walltime $walltime --seeds $SEEDS -- \
        --task chase \
        --set normalize_obs=false --set adaptive_lr=false \
        --set lr=1.25e-4 --set ent_coef=0.001 \
        --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
        --set net=attention \
        "$@" \
        --out runs/14_attention_presets/$label
}

run apf_3v1 2:00:00 --preset apf_3v1 --steps 5e8
run apf_x8_free_3v1 2:00:00 --preset apf_x8_free_3v1 --steps 5e8
run apf_3v1_weak 2:00:00 --preset apf_3v1_weak --steps 5e8
run apf_3v1_weak_v4_a7 2:00:00 --preset apf_3v1_weak --steps 5e8 \
    --set-env weak_max_rpm=4679 --set-env weak_air_resistance_coeff=4.244
run apf_3v1_weak_v4_a7_steps2e9 3:59:00 --preset apf_3v1_weak --steps 2e9 \
    --set-env weak_max_rpm=4679 --set-env weak_air_resistance_coeff=4.244
