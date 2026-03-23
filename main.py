"""
main.py  –  Punto de entrada para entrenar/evaluar PPO o SAC.

Uso:
  python main.py ppo_cartpole_pixel --train
  python main.py ppo_walker2d_pixel --train
  python main.py ppo_walker2d_pixel          # solo eval (carga checkpoint)
"""
import argparse
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PPO / SAC.")
    parser.add_argument("hyperparameters", help="Nombre del set de hiperparámetros (clave en el YAML)")
    parser.add_argument("--train", action="store_true", help="Modo entrenamiento (sin --train = evaluación)")
    args = parser.parse_args()

    config = load_config(args.hyperparameters)
    algo   = config.get("algo", "ppo").lower()

    if algo == "ppo":
        from src.ppo.train import PPOAgent
        agent = PPOAgent(args.hyperparameters)
    elif algo == "sac":
        from src.sac.train import SACAgent
        agent = SACAgent(args.hyperparameters)
    else:
        raise ValueError(f"Algoritmo desconocido: {algo}. Usa 'ppo' o 'sac'.")

    agent.run(is_training=args.train)


if __name__ == "__main__":
    main()