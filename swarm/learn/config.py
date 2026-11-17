from dataclasses import dataclass


@dataclass
class TrainConfig:
    task: str
    preset: str
    seed: int = 0

    num_envs: int = 4096
    num_steps: int = 64
    total_timesteps: int = 100_000_000

    lr: float = 2e-4
    adaptive_lr: bool = True

    gamma: float = 0.998
    gae_lambda: float = 0.95

    num_minibatches: int = 32
    update_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    max_grad_norm: float = 0.5

    generations: int = 0

    net: str = "mlp"
    hidden: tuple = (256, 256)
    activation: str = "tanh"
    init_std: float = 1.0
    reset_std: bool = True

    normalize_reward: bool = True
    normalize_obs: bool = True

    log_every: int = 10
    checkpoint_every: int = 50

    @property
    def num_updates(self):
        return self.total_timesteps // (self.num_steps * self.num_envs)

    @property
    def minibatch_size(self):
        return (self.num_envs * self.num_steps) // self.num_minibatches
