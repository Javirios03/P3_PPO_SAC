import os
import yaml
import torch

DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
CONFIG   = "./configs/hyperparameters.yml"
os.makedirs(RUNS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(hyperparameter_set: str) -> dict:
    """
    Carga la configuración para un experimento.
    Si ya existe una config guardada en runs/<nombre>/config.yml
    (porque se ha reanudado un experimento), la usa.
    Si no, la carga del YAML principal y la persiste.
    """
    run_config_path = os.path.join(RUNS_DIR, hyperparameter_set, "config.yml")

    if os.path.exists(run_config_path):
        with open(run_config_path, "r") as f:
            return yaml.safe_load(f)

    with open(CONFIG, "r") as f:
        all_configs = yaml.safe_load(f)

    if hyperparameter_set not in all_configs:
        available = list(all_configs.keys())
        raise KeyError(
            f"Config '{hyperparameter_set}' no encontrada. "
            f"Disponibles: {available}"
        )

    config = all_configs[hyperparameter_set]
    os.makedirs(os.path.join(RUNS_DIR, hyperparameter_set), exist_ok=True)

    with open(run_config_path, "w") as f:
        yaml.dump(config, f)

    return config