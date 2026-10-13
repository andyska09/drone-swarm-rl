# drone-swarm

A multirotor simulator in JAX, written to train swarm policies with RL on thousands
of parallel environments. The goal is decentralized interception: N drones catch one
agile evader, with no communication between them.

The physics and the control cascade follow
[ctu-mrs/mrs_multirotor_simulator](https://github.com/ctu-mrs/mrs_multirotor_simulator).

## Setup

```bash
conda create -n drone-swarm python=3.12
conda run -n drone-swarm pip install -r requirements.txt
```

## Use

```bash
conda run -n drone-swarm pytest tests/
conda run -n drone-swarm python run/train.py --task circle_swap --preset pair --steps 5e6
conda run -n drone-swarm python run/eval.py runs/<name>
conda run -n drone-swarm python run/replay.py
```

Tasks: `circle_swap` (presets `pair`, `circle8`) and `chase` (preset `default`).

## Layout

- `swarm/sim/` — plant, control cascade, hand-written opponents
- `swarm/envs/` — one file per task
- `swarm/learn/` — PPO, eval, opponent pools
- `run/` — train, eval, replay, plot
- `tools/viewer/` — 3D replay viewer
- `research/` — notes and papers
