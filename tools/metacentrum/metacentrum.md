# Running training on MetaCentrum

`submit.py` sends one `qsub` per job, and `job.sh` runs `run/train.py` on a GPU node.
Every job writes straight into `runs/` in the repo on your home, so there is no
copy-back step on the cluster.

Every job asks for these resources. A flag changes one for that submit.

| resource | default | flag |
|---|---|---|
| CPUs | 2 | `--ncpus` |
| RAM | 32 GB | `--mem` |
| GPUs | 1 | — |
| GPU generation | `compute_80` or newer: A100, A40, A4000, L40, L40s, H100 | `--gpu-cap` |
| time limit | 12 h | `--walltime` |

PBS picks the queue itself: `gpu` up to 24 h, `gpu_long` above that.

## 1. Log in

```
ssh YOUR_USERNAME@skirit.metacentrum.cz
```

The frontends are for editing and submitting only. Never compute on them. The home on
`skirit` is `/storage/brno2/home/YOUR_USERNAME`. Another frontend has another home.

## 2. First-time setup

Clone the repo:

```
git clone https://github.com/andyska09/drone-swarm-rl.git ~/drone-swarm-rl
```

Create the env:

```
conda create -n drone-swarm python=3.12 pip -y
```

Install the pinned packages and the CUDA 12 build of JAX. pip downloads about 3 GB,
which overflows the default temp folder, so give it one on your home:

```
conda activate drone-swarm
mkdir -p ~/pip-tmp
TMPDIR=~/pip-tmp pip install --no-cache-dir -r ~/drone-swarm-rl/requirements.txt "jax[cuda12]==0.11.1"
rm -rf ~/pip-tmp
```

`pip list | grep -iE "jax|nvidia"` must show `jax-cuda12-plugin`, `jax-cuda12-pjrt` and
the `nvidia-...` packages. Without them JAX runs on the CPU.

Check the GPU in a short interactive job:

```
qsub -I -l select=1:ncpus=2:ngpus=1:mem=8gb -l walltime=0:30:00
```

On the GPU node:

```
conda activate /storage/brno2/home/YOUR_USERNAME/.conda/envs/drone-swarm
python -c "import jax; print(jax.devices())"
exit
```

It must print `[CudaDevice(id=0)]`.

## 3. Submit

The code gets to the cluster through git only. Push on the Mac, then on `skirit`:

```
cd ~/drone-swarm-rl
git pull
git log --oneline -1
```

The commit must be the one you pushed. Then submit from `~/drone-swarm-rl`. Everything
after `--` goes to `run/train.py` unchanged.

Check the `qsub` command first. `--dry-run` prints it and sends nothing:

```
python3 tools/metacentrum/submit.py --dry-run -- --task chase --preset default --steps 1.4e9
```

One run:

```
python3 tools/metacentrum/submit.py -- --task circle_swap --preset pair --steps 5e6
```

One job per seed:

```
python3 tools/metacentrum/submit.py --seeds 0 1 2 -- --task chase --preset default --set generations=13 --steps 1.4e9
```

Set the time limit. A pool run of 1.4e9 steps takes 17–45 min, so 1 h is enough:

```
python3 tools/metacentrum/submit.py --walltime 1:00:00 -- --task chase --preset default --set generations=13 --steps 1.4e9
```

Resume a pool run that ran out of time:

```
python3 tools/metacentrum/submit.py -- --resume runs/<name>
```

The job name is the start of the run folder name, for example `chase_default_s0`. A
resume is `resume_<name>`.

## 4. Watch

Your jobs:

```
qstat -u YOUR_USERNAME
```

How busy the GPU is (`gpupercent`):

```
qstat -fw <job id> | grep gpu
```

The newest runs, and the last rows of one:

```
ls -t runs | head
tail -n 3 runs/<name>/metrics.csv
```

The job log lands in `logs/<job id>.OU`, for example `logs/24639435.pbs-m1.metacentrum.cz.OU`,
only when the job ends, because PBS holds it until then. Its top lines show the
arguments, the card, `jax.devices()` and the run folder.

## 5. Copy back

On the Mac, from the repo root:

```
rsync -av YOUR_USERNAME@skirit.metacentrum.cz:~/drone-swarm-rl/runs/ runs/
```

A run folder name holds a time stamp and a hash, so it never clashes with a run made
on the Mac. Then run `run/eval.py`, `run/matrix.py` and `run/replay.py` on the Mac as usual.

## Traps

- **A job runs the code that is in `~/drone-swarm-rl` when it starts, not when you
  submit it.** A `git pull` while a job waits in the queue changes what it runs. A
  running job is safe, because Python reads all the code at the start. The run folder
  records the commit it ran.
- **Only a pool run resumes** (`generations` above 0). A run with `generations=0` that
  hits the time limit is lost, so give it enough `--walltime`.
- **Activate the env by its full path on a GPU node.** A node in another city has
  another `$HOME`, so `conda activate drone-swarm` can fail there. `job.sh` uses
  `$PBS_O_HOME`, the home of the frontend you submitted from.
- **If `jax.devices()` shows `CpuDevice`**, first check that `jax-cuda12-plugin` is
  installed. If it is, run `unset LD_LIBRARY_PATH`, because it can override the CUDA
  libraries that JAX brings.
- **`Disk quota exceeded` from pip** means the temp folder is full, not your home.
  Install on the frontend with `TMPDIR` set, as in step 2.
