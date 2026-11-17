#!/bin/bash
set -euo pipefail

SEEDS="0 1"

run() {
    label=$1
    shift
    python3 tools/rci/submit.py --walltime 2:00:00 --seeds $SEEDS -- \
        --task chase --preset paths_3v1 --steps 5e8 \
        --set gamma=0.99 --set normalize_obs=false \
        --set adaptive_lr=false --set lr=1.25e-4 --set ent_coef=0.003 \
        "$@" \
        --out runs/03_breadth/$label
}

run gamma0.98 --set gamma=0.98
run gamma0.995 --set gamma=0.995
run gae_lambda0.9 --set gae_lambda=0.9
run gae_lambda0.98 --set gae_lambda=0.98
run clip_eps0.1 --set clip_eps=0.1
run clip_eps0.3 --set clip_eps=0.3
run update_epochs2 --set update_epochs=2
run update_epochs8 --set update_epochs=8
run num_minibatches16 --set num_minibatches=16
run num_minibatches64 --set num_minibatches=64
run num_steps32 --set num_steps=32
run num_steps128 --set num_steps=128
run num_envs2048 --set num_envs=2048
run num_envs8192 --set num_envs=8192
run init_std0.3 --set init_std=0.3
run init_std0.5 --set init_std=0.5
run max_grad_norm1.0 --set max_grad_norm=1.0
run max_steps2000 --set-env max_steps=2000
run coll_pp0 --set-env reward.coll_pp=0
