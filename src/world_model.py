import torch

from cfg import Config
from mlp import Mlp


class WorldModel:
    def __init__(self, cfg: Config) -> None:
        self._dynamics_net = Mlp(
            in_dim=cfg.latent_dim + cfg.task_dim + cfg.action_dim,
            layer_sizes=[cfg.layer_size] * cfg.n_layers,
            out_dim=cfg.latent_dim,
        )
        self._reward_net = Mlp(
            in_dim=cfg.latent_dim + cfg.task_dim + cfg.action_dim,
            layer_sizes=[cfg.layer_size] * cfg.n_layers,
            out_dim=1,
        )
        self._done_net = Mlp(
            in_dim=cfg.latent_dim + cfg.task_dim,
            layer_sizes=[cfg.layer_size] * cfg.n_layers,
            out_dim=1,
        )

        def next_state(z: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
            x = torch.cat(z, task)
            return self._dynamics_net.forward(x)

        def reward(z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            x = torch.cat(z, action)
            return self._reward_net.forward(x)

        def is_done(z: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
            x = torch.cat(z, task)
            return self._done_net.forward(x)
