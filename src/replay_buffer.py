import torch
from tensordict import TensorDict
import torchrl.data.replay_buffers as rb

from cfg import Config

class ReplayBuffer:
    def __init__(self, cfg: Config) -> None:
        self._capacity = max(cfg.buffer_size, cfg.min_history)
        self._size = 0
        self._sampler = rb.SliceSampler(
            num_slices=cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
            cache_values=True
        )
        self._buffer = rb.ReplayBuffer(
            storage=rb.LazyTensorStorage(self._capacity, device="cpu"),
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=cfg.batch_size * (cfg.horizon + 1))

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def add(self, transition: TensorDict) -> None:
        pass

    def sample(self) -> tuple[torch.Tensor]:
        pass
