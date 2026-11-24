#!/bin/bash
set -euo pipefail

SEEDS="0 1"

run() {
    label=$1
    shift
    for pursuers in 1 3; do
        preset=apf_${pursuers}v1_weak
        out=runs/12_weak_pursuer/$preset$label
        if [ -d $out ]; then
            echo "skip $out: it has runs"
            continue
        fi
        python3 tools/rci/submit.py --walltime 1:00:00 --seeds $SEEDS -- \
            --task chase --preset $preset --steps 5e8 \
            --set normalize_obs=false --set adaptive_lr=false \
            --set lr=1.25e-4 --set ent_coef=0.001 \
            --set gamma=0.995 --set clip_eps=0.1 --set num_steps=128 \
            "$@" \
            --out $out
    done
}

run ""
run _v6_a13 --set-env weak_air_resistance_coeff=3.773
run _v4_a7 --set-env weak_max_rpm=4679 --set-env weak_air_resistance_coeff=4.244
