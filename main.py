import argparse

from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="Name of the hyperparameter set")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()
    
    config = load_config(args.hyperparameters)
    algo = config.get("algo", "ppo").lower()

    if algo == "ppo":
        from src.ppo.train import PPOAgent
        agent = PPOAgent(args.hyperparameters)
    elif algo == "sac":
        from src.sac.train import SACAgent
        agent = SACAgent(args.hyperparameters)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)


if __name__ == "__main__":
    main()
