import argparse

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="Name of the hyperparameter set")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    # Load config to determine whether to use Rainbow or vanilla DQN
    config = load_config(args.hyperparameters)

    rainbow_flags = [
        config.get("enable_prioritized_replay", False),
        config.get("enable_noisy_nets", False),
        config.get("enable_distributional", False),
        config.get("enable_n_step", False),
    ]

    if any(rainbow_flags):
        from src.rainbow.train import RainbowAgent
        agent = RainbowAgent(args.hyperparameters)
    else:
        from src.dqn.train import DQNAgent
        agent = DQNAgent(args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)


if __name__ == "__main__":
    main()
