# main.py
import argparse
from src.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Train or test PPO/SAC agent.")
    parser.add_argument("hyperparameters", help="Name of the hyperparameter set")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.hyperparameters)
    algo = config.get("algo", "ppo")   # "ppo" o "sac"

    if algo == "ppo":
        from src.ppo.train import PPOAgent
        agent = PPOAgent(args.hyperparameters)
    elif algo == "sac":
        from src.sac.train import SACAgent
        agent = SACAgent(args.hyperparameters)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    agent.run(is_training=args.train)

if __name__ == "__main__":
    main()
