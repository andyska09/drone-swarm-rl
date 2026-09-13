"""PPO hyperparameters and run settings."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    task: str = "a_to_b"
    preset: str = "default"
    seed: int = 0

    num_envs: int = 4096
    num_steps: int = 64
    total_timesteps: int = 100_000_000

    lr: float = 3e-4
    anneal_lr: bool = True

    gamma: float = 0.998
    gae_lambda: float = 0.95

    num_minibatches: int = 32
    update_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5

    hidden: tuple = (256, 256)
    activation: str = "tanh"
    # A smaller initial standard deviation reduces action clipping.
    init_log_std: float = -0.5

    normalize_reward: bool = True

    log_every: int = 10
    checkpoint_every: int = 50

    @property
    def num_updates(self):
        return self.total_timesteps // (self.num_steps * self.num_envs)

    @property
    def minibatch_size(self):
        return (self.num_envs * self.num_steps) // self.num_minibatches
