import os
import yaml
import torch

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
