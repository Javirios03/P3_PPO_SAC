import os
import yaml
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── Config / constants ──────────────────────────────────────────────

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
CONFIG = "./configs/hyperparameters.yml"
os.makedirs(RUNS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(hyperparameter_set):
    """Load config from runs dir (if resuming) or from main configs file."""
    config_file = os.path.join(RUNS_DIR, hyperparameter_set, "config.yml")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    else:
        with open(CONFIG, "r") as f:
            all_config = yaml.safe_load(f)
            config = all_config[hyperparameter_set]
            os.makedirs(os.path.join(RUNS_DIR, hyperparameter_set), exist_ok=True)
            with open(config_file, "w") as f:
                yaml.dump(config, f)
            return config


# ── Plotting ────────────────────────────────────────────────────────

def save_graph(graph_file, rewards_per_episode, actor_losses, critic_losses, entropies):
    fig = plt.figure(1, figsize=(15, 4))

    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : (x + 1)])

    plt.subplot(141)
    plt.ylabel("Mean Rewards")
    plt.plot(mean_rewards)

    plt.subplot(142)
    plt.ylabel("Actor Loss")
    plt.plot(actor_losses)

    plt.subplot(143)
    plt.ylabel("Critic Loss")
    plt.plot(critic_losses)

    plt.subplot(144)
    plt.ylabel("Entropy")
    plt.plot(entropies)

    plt.subplots_adjust(wspace=0.5, hspace=1.0)
    fig.savefig(graph_file)
    plt.close(fig)
