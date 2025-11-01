import torch
import torch.nn as nn


class NnBlock(nn.Module):
    """
    A basic neural network block consisting of:
    Linear -> LayerNorm -> Mish

    Args:
    n_in : int      The number of input features
    n_out: int      The number of output features
    """

    def __init__(self, n_in: int, n_out: int) -> None:
        super(NnBlock, self).__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.layer_norm = nn.LayerNorm(n_out)
        self.mish = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.mish(x)

        return x
