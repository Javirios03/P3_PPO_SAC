
# DQN / Pixel-DQN in MuJoCo (Gymnasium)

This repository trains **DQN-style agents** on MuJoCo tasks using Gymnasium, with two observation modes:

- **state**: classic low-dimensional state vectors.
- **pixel**: visual observations from rendered RGB frames (resized + frame-stacked).

Training/evaluation is driven by presets in `configs/hyperparameters.yml`.

## Theoretical background (DQN)

The core idea of **Deep Q-Learning** is to approximate the optimal action-value function $Q^*(s, a)$ with a neural network $Q(s, a; \theta)$.

### Bellman optimality equation

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

Where:

- $s, s'$ are current/next observations (state vectors or pixel stacks).
- $a, a'$ are actions from a **discrete** action space.
- $r$ is the reward.
- $\gamma \in [0, 1]$ is the discount factor.

### TD target + loss

With a target network ($\theta^-$), the TD target is:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

and the DQN is trained by minimizing the mean-squared TD error:

$$L(\theta) = \mathbb{E}\left[(y - Q(s, a; \theta))^2\right]$$

### Stability mechanisms

This implementation follows the standard stabilizers:

- **Experience replay**: store transitions $(s, a, r, s', done)$ and sample mini-batches.
- **Target network**: periodically copy weights to make the target less non-stationary.

### Pixels: why the first Conv2D uses 12 channels

In pixel mode the env produces observations shaped like `[stack, H, W, C]`.
With `stack_size=4` and RGB (`C=3`), the effective input channels are $4 \times 3 = 12$.
That is why the CNN starts with `Conv2d(12, ...)`.

## Rainbow DQN extensions

On top of vanilla DQN, the following improvements can be toggled independently via `configs/hyperparameters.yml`:

| Flag | Technique |
|---|---|
| `enable_double_dqn` | Double DQN — decouples action selection from evaluation to reduce overestimation |
| `enable_dueling_dqn` | Dueling architecture — separate value and advantage streams |
| `enable_prioritized_replay` | Prioritized Experience Replay (PER) — sample important transitions more often |
| `enable_noisy_nets` | NoisyNets — learned exploration via stochastic linear layers |
| `enable_distributional` | C51 — categorical distributional RL |
| `enable_n_step` | Multi-step returns — accumulate rewards over $n$ steps |

When **any** Rainbow flag is enabled the CLI automatically uses `RainbowAgent`; otherwise it uses the lightweight `DQNAgent`.

## Repository structure

```
.
├── main.py                    # CLI entry point (train / evaluate)
├── pyproject.toml             # Dependencies and project metadata
├── configs/
│   └── hyperparameters.yml    # Run presets (env, obs type, hyperparameters)
├── runs/                      # Generated outputs (one folder per run)
│   └── <run_name>/
│       ├── config.yml         # Frozen copy of the configuration used
│       ├── best_model.pt      # Best model weights (by episodic return)
│       ├── checkpoint.pt      # Periodic checkpoint (resumable)
│       ├── training.log       # Plain-text training log
│       ├── graph.png          # Reward + epsilon plot
│       └── tensorboard/       # TensorBoard logs (if generated)
├── src/
│   ├── __init__.py
│   ├── config.py              # Config loading, paths, device selection
│   ├── networks.py            # DQN (MLP), Pixel_DQN (CNN), NoisyLinear
│   ├── buffer.py              # Experience Replay (circular buffer)
│   ├── wrappers.py            # Env wrappers + factory functions
│   ├── utils.py               # Plotting helpers (save_graph)
│   ├── dqn/
│   │   ├── __init__.py
│   │   └── train.py           # DQNAgent — vanilla DQN training & evaluation
│   └── rainbow/
│       ├── __init__.py
│       ├── buffer.py          # PrioritizedExperienceReplay
│       └── train.py           # RainbowAgent — Rainbow DQN training & evaluation
├── notebooks/                 # Exploration / debug notebooks
└── Report/                    # LaTeX report
```

### What gets saved to `runs/`?

Each preset creates its own folder under `runs/<preset_name>/` containing:

| File | Description |
|---|---|
| `config.yml` | Frozen copy of the hyperparameters used for the run |
| `best_model.pt` | Model weights with the highest episodic return so far |
| `checkpoint.pt` | Full training state (model, optimizer, epsilon, episode, rewards) — saved every 100 episodes, allows resuming |
| `training.log` | Timestamped log entries (new bests, checkpoints) |
| `graph.png` | Plot with mean reward (100-episode window) and epsilon decay |

Training is **automatically resumable**: if `runs/<preset>/checkpoint.pt` exists, the agent restores its full state and continues from the last saved episode.

## Installation

### Requirements

- Python >= 3.10
- MuJoCo via `gymnasium[mujoco]` (installed as a dependency)

### Setup with `uv`

```bash
git clone https://github.com/pepert03/DQN-Rainbow-Pixel-Control
cd DQN-Rainbow-Pixel-Control

uv sync
```

## Usage

All commands are run from the repository root via `main.py`.

### 1) Choose a preset

Presets are defined in `configs/hyperparameters.yml`. Each preset specifies the environment, observation type, and all hyperparameters. Available presets (out of the box):

| Preset | Environment | Obs | Rainbow |
|---|---|---|---|
| `walker2d` | Walker2d-v5 | pixel | No |
| `rainbow_pixel_walker2d` | Walker2d-v5 | pixel | Yes |
| `rainbow_states_walker2d` | Walker2d-v5 | state | Yes |
| `rainbow_cartpole` | CartPole-v1 | pixel | Yes |

### 2) Training

```bash
# Vanilla DQN
python main.py walker2d --train

# Rainbow DQN
python main.py rainbow_pixel_walker2d --train

# With uv
uv run python main.py walker2d --train
```

### 3) Evaluation (with rendering)

Loads the best saved model from `runs/<preset>/best_model.pt` and renders the agent:

```bash
python main.py walker2d

uv run python main.py rainbow_states_walker2d
```

Notes:

- In **pixel** mode, the wrapper creates the env with `render_mode="rgb_array"` to obtain frames, and uses an OpenCV window for live preview.
- In **state** mode, the env uses `render_mode="human"` when rendering.