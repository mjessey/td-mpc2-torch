# TD-MPC2 Robot Soccer Agent

## TD-MPC2 Replay Buffer

A PyTorch implementation of a replay buffer for TD-MPC2 (Temporal Difference Model Predictive Control) for training robot soccer agents.

## Features

- **Smart Memory Management**: Automatically selects CUDA or CPU storage based on available memory
- **Episode-Based Storage**: Stores complete episodes and samples contiguous sequences
- **Lazy Initialization**: Buffer is initialized on first episode for optimal memory allocation
- **Efficient Sampling**: Uses TorchRL's SliceSampler for fast contiguous sequence sampling

## Quick Start

```python
from src.replay_buffer import ReplayBuffer
from src.cfg import read_cfg
from tensordict import TensorDict
import torch

# Load configuration
cfg = read_cfg()

# Create replay buffer
buffer = ReplayBuffer(cfg)

# Add an episode (TensorDict with shape (episode_length, ...))
episode = TensorDict({
    'obs': torch.randn(100, 64),      # 100 steps, 64-dim observation
    'action': torch.randn(100, 8),    # 100 steps, 8-dim action
    'reward': torch.randn(100),       # 100 rewards
    'done': torch.zeros(100),         # done flags
}, batch_size=[100])

buffer.add(episode)

# Sample when ready
if buffer.ready():
    obs, action, reward, done = buffer.sample()
    # obs: (horizon+1, batch_size, obs_dim)
    # action: (horizon, batch_size, action_dim)
    # reward: (horizon, batch_size, 1)
    # done: (horizon, batch_size, 1)
```

## API Reference

### `ReplayBuffer(cfg: Config)`

Initialize the replay buffer.

**Properties:**
- `capacity`: Maximum number of transitions
- `num_eps`: Number of episodes stored

**Methods:**

#### `add(td: TensorDict) -> int`
Add a single episode to the buffer.
- **Args**: `td` - TensorDict with shape (episode_length, ...)
- **Returns**: Number of episodes in buffer

#### `load(td: TensorDict) -> int`
Load multiple episodes at once (more efficient for batch loading).
- **Args**: `td` - TensorDict with shape (num_episodes, episode_length, ...)
- **Returns**: Number of episodes in buffer

#### `sample() -> tuple[Tensor, ...]`
Sample a batch of contiguous sequences.
- **Returns**: Tuple of (obs, action, reward, done) tensors

#### `ready() -> bool`
Check if buffer is ready for sampling.
- **Returns**: True if buffer has at least one episode

## Configuration

Edit `cfg/default.toml`:

```toml
[config]
buffer_size = 1_000_000    # Maximum transitions
batch_size = 256           # Sequences per batch
min_history = 100_000      # Minimum before training
horizon = 3                # Planning horizon
```

## Memory Management

The buffer automatically:
1. Estimates memory requirements from the first episode
2. Checks available CUDA memory
3. Uses CUDA if `2.5 × required_memory < available_memory`
4. Falls back to CPU otherwise
