# PPO / SAC on Walker2D with pixels observations

This repository trains **PPO and SAC agents** on MuJoCo tasks using Gymnasium, with two observation modes:

- **state**: classic low-dimensional state vectors.
- **pixel**: visual observations from rendered RGB frames (resized + frame-stacked).

Training/evaluation is driven by presets in `configs/hyperparameters.yml`.

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

| File            | Description                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------------- |
| `config.yml`    | Frozen copy of the hyperparameters used for the run                                                           |
| `best_model.pt` | Model weights with the highest episodic return so far                                                         |
| `checkpoint.pt` | Full training state (model, optimizer, epsilon, episode, rewards) — saved every 100 episodes, allows resuming |
| `training.log`  | Timestamped log entries (new bests, checkpoints)                                                              |
| `graph.png`     | Plot with mean reward (100-episode window) and epsilon decay                                                  |

Training is **automatically resumable**: if `runs/<preset>/checkpoint.pt` exists, the agent restores its full state and continues from the last saved episode.

## Installation

### Requirements

- Python >= 3.13
- MuJoCo via `gymnasium[mujoco]` (installed as a dependency)

### Setup with `uv`

```bash
git clone https://github.com/Javirios03/P3_PPO_SAC.git
cd P3_PPO_SAC

uv sync
```

## Usage

All commands are run from the repository root via `main.py`.

### 1) Choose a preset

Presets are defined in `configs/hyperparameters.yml`. Each preset specifies the environment, observation type, and all hyperparameters. Available presets (out of the box):

| Preset                    | Environment | Obs   | Rainbow |
| ------------------------- | ----------- | ----- | ------- |
| `walker2d`                | Walker2d-v5 | pixel | No      |
| `rainbow_pixel_walker2d`  | Walker2d-v5 | pixel | Yes     |
| `rainbow_states_walker2d` | Walker2d-v5 | state | Yes     |
| `rainbow_cartpole`        | CartPole-v1 | pixel | Yes     |

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
