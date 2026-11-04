# Running training on RCI

`submit.py` sends one `sbatch` per job, and `job.sh` runs `run/train.py` on a GPU node.
Every job writes straight into `runs/` in the repo on `/mnt/personal`, so there is no
copy-back step on the cluster.

Every job asks for these resources. A flag changes one for that submit.

| resource | default | flag |
|---|---|---|
| CPUs | 2 | `--ncpus` |
| RAM | `32G` | `--mem` |
| GPUs | 1 A100 40 GB | — |
| time limit | 12 h | `--walltime`, as `H:MM:SS` |

A partition is a job queue. `submit.py` picks it from the time limit. `sinfo` prints the
limits that hold today. The `amd` in the names is the CPU of the node. The GPU is NVIDIA.

| time limit | partition |
|---|---|
| under 4 h | `amdgpufast` |
| under 24 h | `amdgpu` |
| under 72 h | `amdgpulong` |
| under 504 h | `amdgpuextralong` |

## 1. Log in

The username is your CTU username. If you have no RCI password yet, set it on the
[profile page](https://login.rci.cvut.cz/secure/).

A password works only on `login1`, so send your SSH key there. On the Mac:

```
ssh-copy-id YOUR_USERNAME@login1.rci.cvut.cz
```

If it finds no key, run `ssh-keygen -t ed25519` first. Every login node shares your home,
so the key now works on all of them. Log in to `login3`, which matches the AMD nodes that
run the jobs:

```
ssh YOUR_USERNAME@login3.rci.cvut.cz
```

The login nodes are for editing and submitting only. Never compute on them, and do not
connect VS Code to `login3`. Keep the repo and the env in `/mnt/personal/$USER`, not in
your home. It is faster and bigger, and every node sees it.

## 2. First-time setup

Check your SLURM account, the group that pays for your jobs:

```
sacctmgr -n show assoc where user=$USER format=account,qos%30
```

If this prints nothing, ask your supervisor which account to use. If the account is not
your default one, `sbatch` stops with `Invalid account`. Then make it the default:

```
echo 'export SBATCH_ACCOUNT=YOUR_ACCOUNT SLURM_ACCOUNT=YOUR_ACCOUNT' >> ~/.bashrc
source ~/.bashrc
```

Clone the repo:

```
git clone https://github.com/andyska09/drone-swarm-rl.git /mnt/personal/$USER/drone-swarm-rl
```

Create the env. The wiki advises against conda, and no module has our pinned `jax`, so the
env is a `venv` on top of the `Python` module:

```
ml Python/3.12.3-GCCcore-13.3.0
python -m venv /mnt/personal/$USER/drone-swarm-env
source /mnt/personal/$USER/drone-swarm-env/bin/activate
```

Install the pinned packages and the CUDA 12 build of JAX. pip downloads about 3 GB, so
give it a temp folder on `/mnt/personal`:

```
mkdir -p /mnt/personal/$USER/pip-tmp
TMPDIR=/mnt/personal/$USER/pip-tmp pip install --no-cache-dir -r /mnt/personal/$USER/drone-swarm-rl/requirements.txt "jax[cuda12]==0.11.1"
rm -rf /mnt/personal/$USER/pip-tmp
```

`pip list | grep -iE "jax|nvidia"` must show `jax-cuda12-plugin`, `jax-cuda12-pjrt` and
the `nvidia-...` packages. Without them JAX runs on the CPU.

Check the GPU in a short interactive job:

```
srun -p amdgpufast --gres=gpu:a100:1 --cpus-per-task=2 --mem=8G --time=0:30:00 --mpi=none --pty bash -i
```

On the GPU node:

```
ml Python/3.12.3-GCCcore-13.3.0
source /mnt/personal/$USER/drone-swarm-env/bin/activate
nvidia-smi -L
python -c "import jax; print(jax.devices())"
exit
```

`nvidia-smi -L` prints the card. The last command must print `[CudaDevice(id=0)]`.

## 3. Submit

The code gets to the cluster through git only. Push on the Mac, then on `login3`:

```
cd /mnt/personal/$USER/drone-swarm-rl
git pull
git log --oneline -1
```

The commit must be the one you pushed. Then submit from the repo. Everything after `--`
goes to `run/train.py` unchanged.

Check the `sbatch` command first. `--dry-run` prints it and sends nothing:

```
python3 tools/rci/submit.py --dry-run -- --task chase --preset default --steps 1.4e9
```

One run:

```
python3 tools/rci/submit.py -- --task circle_swap --preset pair --steps 5e6
```

One job per seed:

```
python3 tools/rci/submit.py --seeds 0 1 2 -- --task chase --preset default --set generations=13 --steps 1.4e9
```

Set the time limit:

```
python3 tools/rci/submit.py --walltime 1:00:00 -- --task chase --preset default --set generations=13 --steps 1.4e9
```

Resume a pool run that ran out of time:

```
python3 tools/rci/submit.py -- --resume runs/<name>
```

The job name is the start of the run folder name, for example `chase_default_s0`. A
resume is `resume_<name>`.

## 4. Watch

Your jobs:

```
squeue -u $USER
```

How busy the GPU is:

```
srun --jobid=<job id> --overlap --mpi=none nvidia-smi
```

The newest runs, and the last rows of one:

```
ls -t runs | head
tail -n 3 runs/<name>/metrics.csv
```

The job log is `logs/<job id>.out`. SLURM writes it while the job runs, so you can follow
it with `tail -f`. Its top lines show the arguments, the card, `jax.devices()` and the run
folder.

## 5. Copy back

On the Mac, from the repo root:

```
rsync -av YOUR_USERNAME@login3.rci.cvut.cz:/mnt/personal/YOUR_USERNAME/drone-swarm-rl/runs/ runs/
```

A run folder name holds a time stamp and a hash, so it never clashes with a run made
on the Mac. Then run `run/eval.py`, `run/matrix.py` and `run/replay.py` on the Mac as usual.

## Traps

- **A job runs the code that is in the repo when it starts, not when you submit it.** A
  `git pull` while a job waits in the queue changes what it runs. A running job is safe,
  because Python reads all the code at the start. The run folder records the commit it
  ran.
- **Only a pool run resumes** (`generations` above 0). A run with `generations=0` that
  hits the time limit is lost, so give it enough `--walltime`.
- **If `jax.devices()` shows `CpuDevice`**, first check that `jax-cuda12-plugin` is
  installed. If it is, run `ml` to list the loaded modules. Unload any `CUDA` module,
  because it can override the CUDA libraries that JAX brings.
- **`cuda_executor driver version` warnings are harmless** if `jax.devices()` shows
  `CudaDevice`. They say that the driver and the CUDA from pip differ.
- **`git pull` can stop on an untracked file and leave the old code.** That is why step 3
  checks `git log --oneline -1` after every pull.
