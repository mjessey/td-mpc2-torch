import torch
import torch.nn as nn

from nn_block import NnBlock
from cfg import Config


class Encoder(nn.Module):
    """
    The encoder / representation network. Converts observations into latent states

    Args:
    cfg: Config     The config object storing hyper-parameters and model details
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self._nn = nn.ModuleList()

        if cfg.encoder_depth == 1:
            self._nn.append(NnBlock(cfg.n_features, cfg.latent_dim))
        else:
            self._nn.append(NnBlock(cfg.n_features, cfg.encoder_width))
            for i in range(cfg.encoder_depth - 2):
                self._nn.append(NnBlock(cfg.encoder_width, cfg.encoder_width))

            self._nn.append(NnBlock(cfg.encoder_width, cfg.latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self._nn:
            x = block(x)

        return x
