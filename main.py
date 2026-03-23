"""
main.py — Punto de entrada para entrenar o evaluar agentes PPO / SAC.

Uso:
  # Entrenamiento
  python main.py ppo_cartpole_pixel --train
  python main.py ppo_walker2d_pixel --train

  # Evaluación (carga checkpoint_final.pt y renderiza)
  python main.py ppo_cartpole_pixel

  # Evaluación sin ventana de render (útil en servidor sin display)
  python main.py ppo_cartpole_pixel --no-render
"""
import argparse
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Entrena o evalúa un agente PPO/SAC.")
    parser.add_argument("hyperparameters",
                        help="Nombre del set de hiperparámetros (clave en hyperparameters.yml)")
    parser.add_argument("--train", action="store_true",
                        help="Modo entrenamiento. Sin este flag → evaluación.")
    parser.add_argument("--no-render", action="store_true",
                        help="Desactiva el render visual durante la evaluación.")
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
        raise ValueError(f"Algoritmo desconocido: '{algo}'. Usa 'ppo' o 'sac'.")

    render = not args.no_render
    agent.run(is_training=args.train, render=render)


if __name__ == "__main__":
    main()