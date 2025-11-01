import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer as TorchRLReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

from .cfg import Config


class ReplayBuffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    
    Uses CUDA memory if available, and CPU memory otherwise.
    The buffer stores episodes and samples contiguous sequences for training.
    """
    
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._capacity = cfg.buffer_size
        self._horizon = cfg.horizon
        
        # SliceSampler for contiguous sequence sampling
        self._sampler = SliceSampler(
            num_slices=cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
            cache_values=False,
        )
        
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0
        self._buffer = None
        self._storage_device = None

    @property
    def capacity(self) -> int:
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self) -> int:
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage: LazyTensorStorage) -> TorchRLReplayBuffer:
        """
        Reserve a buffer with the given storage.
        
        Args:
            storage: LazyTensorStorage instance
            
        Returns:
            ReplayBuffer instance
        """
        return TorchRLReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size,
        )

    def _init(self, td: TensorDict) -> TorchRLReplayBuffer:
        """
        Initialize the replay buffer. Use the first episode to estimate storage requirements.
        
        Args:
            td: First episode TensorDict
            
        Returns:
            Initialized ReplayBuffer
        """
        print(f'Buffer capacity: {self._capacity:,}')
        
        # Calculate storage requirements
        bytes_per_step = sum([
            (v.numel() * v.element_size() if not isinstance(v, TensorDict)
             else sum([x.numel() * x.element_size() for x in v.values()]))
            for v in td.values()
        ]) / len(td)
        total_bytes = bytes_per_step * self._capacity
        print(f'Storage required: {total_bytes / 1e9:.2f} GB')
        
        # Heuristic: decide whether to use CUDA or CPU memory
        if torch.cuda.is_available():
            mem_free, _ = torch.cuda.mem_get_info()
            storage_device = 'cuda:0' if 2.5 * total_bytes < mem_free else 'cpu'
        else:
            storage_device = 'cpu'
        
        print(f'Using {storage_device.upper()} memory for storage.')
        self._storage_device = torch.device(storage_device)
        
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device)
        )

    def add(self, td: TensorDict) -> int:
        """
        Add an episode to the buffer.
        
        Args:
            td: TensorDict containing episode data with keys:
                - 'obs': observations (T, obs_dim)
                - 'action': actions (T, action_dim)
                - 'reward': rewards (T,)
                - 'done': done flags (T,)
                
        Returns:
            Number of episodes in buffer
        """
        # Add episode ID to all transitions in the episode
        td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
        
        # Initialize buffer on first episode
        if self._num_eps == 0:
            self._buffer = self._init(td)
        
        # Add episode to buffer
        self._buffer.extend(td)
        self._num_eps += 1
        
        return self._num_eps

    def load(self, td: TensorDict) -> int:
        """
        Load a batch of episodes into the buffer. This is useful for loading data from disk,
        and is more efficient than adding episodes one by one.
        
        Args:
            td: TensorDict with shape (num_episodes, episode_length, ...)
            
        Returns:
            Number of episodes in buffer
        """
        num_new_eps = len(td)
        episode_idx = torch.arange(self._num_eps, self._num_eps + num_new_eps, dtype=torch.int64)
        td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
        
        # Initialize buffer on first load
        if self._num_eps == 0:
            self._buffer = self._init(td[0])
        
        # Flatten and add to buffer
        td = td.reshape(td.shape[0] * td.shape[1])
        self._buffer.extend(td)
        self._num_eps += num_new_eps
        
        return self._num_eps

    def _prepare_batch(self, td: TensorDict) -> tuple[torch.Tensor, ...]:
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        
        Args:
            td: Sampled TensorDict with shape (horizon+1, batch_size, ...)
            
        Returns:
            Tuple of (obs, action, reward, done) tensors
        """
        # Select relevant keys and move to device
        td = td.select("obs", "action", "reward", "done", strict=False).to(
            self._device, non_blocking=True
        )
        
        # Extract and prepare tensors
        obs = td.get('obs').contiguous()
        action = td.get('action')[1:].contiguous()
        reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
        
        # Handle done flag
        done = td.get('done', None)
        if done is not None:
            done = done[1:].unsqueeze(-1).contiguous()
        else:
            done = torch.zeros_like(reward)
        
        return obs, action, reward, done

    def sample(self) -> tuple[torch.Tensor, ...]:
        """
        Sample a batch of subsequences from the buffer.
        
        Returns:
            Tuple of (obs, action, reward, done) tensors:
                - obs: (horizon+1, batch_size, obs_dim)
                - action: (horizon, batch_size, action_dim)
                - reward: (horizon, batch_size, 1)
                - done: (horizon, batch_size, 1)
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Add at least one episode first.")
        
        # Sample and reshape: (batch_size * (horizon+1),) -> (batch_size, horizon+1) -> (horizon+1, batch_size)
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        
        return self._prepare_batch(td)
    
    def ready(self) -> bool:
        """
        Check if the buffer has enough samples for training.
        
        Returns:
            True if buffer is initialized and has at least one episode
        """
        return self._buffer is not None and self._num_eps > 0
