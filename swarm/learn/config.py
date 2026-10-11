"""PPO hyperparameters and run settings."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    task: str
    preset: str
    seed: int = 0

    num_envs: int = 4096
    num_steps: int = 64
    total_timesteps: int = 100_000_000

    # The start. After that, each role's lr follows its KL (ppo.adapt_lr).
    lr: float = 2e-4

    gamma: float = 0.998
    gae_lambda: float = 0.95

    num_minibatches: int = 32
    update_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    max_grad_norm: float = 0.5

    # Above 0, chase trains against pools of frozen opponents: generation 0, then
    # this many more (pool.py). At 0 both sides learn at once.
    generations: int = 0

    hidden: tuple = (256, 256)
    activation: str = "tanh"
    # The std a new policy starts at, before it learns one per state.
    init_std: float = 0.6

    normalize_reward: bool = True

    log_every: int = 10
    checkpoint_every: int = 50

    @property
    def num_updates(self):
        return self.total_timesteps // (self.num_steps * self.num_envs)

    @property
    def minibatch_size(self):
        return (self.num_envs * self.num_steps) // self.num_minibatches
