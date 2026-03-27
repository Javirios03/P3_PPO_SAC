import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import gymnasium as gym
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter

from src.utils import load_config, device, RUNS_DIR, DATE_FORMAT, save_graph
from src.wrappers import make_env, make_vec_env

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Conv output size helper
# ---------------------------------------------------------------------------
def _conv_out_size(obs_shape):
    dummy = torch.zeros(1, *obs_shape)
    net = nn.Sequential(
        nn.Conv2d(obs_shape[0], 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten()
    )
    with torch.no_grad():
        return net(dummy).shape[1]


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        obs_shape = obs_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = _conv_out_size(obs_shape)

        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.is_discrete:
            n_actions = action_space.n
            self.actor = nn.Sequential(
                nn.Linear(conv_out, 512), nn.ReLU(),
                nn.Linear(512, n_actions)
            )
        else:
            n_actions = action_space.shape[0]
            self.actor_mu = nn.Sequential(
                nn.Linear(conv_out, 512), nn.ReLU(),
                nn.Linear(512, n_actions),
                nn.Tanh()
            )
            self.log_std = nn.Parameter(torch.zeros(n_actions))

        self.critic = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        if self.is_discrete:
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor_mu[-2].weight, gain=0.01)

    def forward(self, obs):
        x = obs / 255.0
        x = self.conv(x)
        if self.is_discrete:
            dist = Categorical(logits=self.actor(x))
        else:
            mu = self.actor_mu(x)
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        return dist, self.critic(x)

    def get_value(self, obs):
        x = obs / 255.0
        return self.critic(self.conv(x))


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------
class PPOAgent:

    def __init__(self, hyperparameter_set):
        config = load_config(hyperparameter_set)
        self.config = config
        self.hyperparameter_set = hyperparameter_set

        self.env_id = config["env_id"]
        self.obs_type = config.get("obs", "pixel")
        self.num_envs = int(config["num_envs"])
        self.rollout_steps = int(config["rollout_steps"])
        self.update_epochs = int(config["update_epochs"])
        self.batch_size = int(config["batch_size"])
        self.gamma = float(config["gamma"])
        self.gae_lambda = float(config["gae_lambda"])
        self.clip_coef = float(config["clip_coef"])
        self.vf_coef = float(config["vf_coef"])
        self.ent_coef = float(config["ent_coef"])
        self.learning_rate = float(config["learning_rate"])
        self.max_grad_norm = float(config["max_grad_norm"])
        self.env_kwargs = config.get("env_kwargs", {})

        # Path to Run info
        self.LOG_FILE = os.path.join(RUNS_DIR, self.hyperparameter_set, "training.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, self.hyperparameter_set, "best_model.pt")
        self.CHECKPOINT_FILE = os.path.join(RUNS_DIR, self.hyperparameter_set, "checkpoint.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, self.hyperparameter_set, "graph.png")
        self.TB_DIR = os.path.join(RUNS_DIR, self.hyperparameter_set, "tensorboard")

    def load_model(self, is_training=True, render=False):
        if is_training:
            envs = make_vec_env(self.env_id, self.obs_type, self.num_envs, env_kwargs=self.env_kwargs)
            obs_space = envs.single_observation_space
            action_space = envs.single_action_space
        else:
            envs = make_env(self.env_id, self.obs_type, render=render, seed=999, env_kwargs=self.env_kwargs)
            obs_space = envs.observation_space
            action_space = envs.action_space

        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        policy = ActorCritic(obs_space, action_space).to(device)
        print(f"[PPO] Parameters: {sum(p.numel() for p in policy.parameters()):,}")

        optimizer = None
        if is_training:
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate, eps=1e-5)

        start_episode = 0
        best_reward = float("-inf")
        rewards_per_episode = []
        actor_losses = []
        critic_losses = []
        entropies = []
        global_step = 0
        ep_reward_buf = np.zeros(self.num_envs) if is_training else None

        if is_training:
            if os.path.exists(self.CHECKPOINT_FILE):
                checkpoint = torch.load(self.CHECKPOINT_FILE, map_location=device, weights_only=False)
                policy.load_state_dict(checkpoint["policy_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_episode = int(checkpoint.get("episode", 0)) + 1
                best_reward = float(checkpoint.get("best_reward", best_reward))
                rewards_per_episode = list(checkpoint.get("rewards_per_episode", []))
                actor_losses = list(checkpoint.get("actor_losses", []))
                critic_losses = list(checkpoint.get("critic_losses", []))
                entropies = list(checkpoint.get("entropies", []))
                global_step = int(checkpoint.get("global_step", 0))
                print(f"Resumed from episode {start_episode - 1} | global_step={global_step}")
            else:
                print("Starting training from scratch.")
        else:
            if os.path.exists(self.MODEL_FILE):
                policy.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
                policy.eval()
                print(f"Loaded model weights from: {self.MODEL_FILE}")

        return (
            envs, policy, optimizer,
            rewards_per_episode, actor_losses, critic_losses, entropies,
            best_reward, start_episode, global_step, ep_reward_buf,
        )

    def save_model(self, policy, optimizer, episode_reward, episode, best_reward,
                   rewards_per_episode, actor_losses, critic_losses, entropies, global_step):
        if episode_reward > best_reward:
            log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward - best_reward) / abs(best_reward) * 100 if best_reward != float('-inf') else 0:+.1f}%) at episode {episode}, saving model..."
            print(log_message)
            with open(self.LOG_FILE, "a") as file:
                file.write(log_message + "\n")
            torch.save(policy.state_dict(), self.MODEL_FILE)

        elif episode % 100 == 0:
            checkpoint = {
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode": episode,
                "best_reward": best_reward,
                "rewards_per_episode": rewards_per_episode,
                "actor_losses": actor_losses,
                "critic_losses": critic_losses,
                "entropies": entropies,
                "global_step": global_step,
            }
            if os.path.exists(self.CHECKPOINT_FILE):
                os.remove(self.CHECKPOINT_FILE)
            torch.save(checkpoint, self.CHECKPOINT_FILE)
            log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Checkpoint saved at episode {episode}"
            print(log_message)
            with open(self.LOG_FILE, "a") as file:
                file.write(log_message + "\n")

    def run(self, is_training=True, render=False):
        writer = None
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, "a") as file:
                file.write(log_message + "\n")

            writer = SummaryWriter(log_dir=self.TB_DIR)

        # Best eval video tracking
        best_eval_reward = float("-inf")
        if not is_training:
            eval_dir = os.path.join(RUNS_DIR, self.hyperparameter_set, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            reward_file = os.path.join(eval_dir, "best_reward.txt")
            video_file = os.path.join(eval_dir, "best.mp4")
            if os.path.exists(reward_file):
                with open(reward_file, "r") as f:
                    best_eval_reward = float(f.read().strip())
                print(f"Previous best eval reward: {best_eval_reward:.1f}")

        (
            envs, policy, optimizer,
            rewards_per_episode, actor_losses, critic_losses, entropies,
            best_reward, start_episode, global_step, ep_reward_buf,
        ) = self.load_model(is_training=is_training, render=render)

        if is_training:
            self._train_loop(
                envs, policy, optimizer, writer,
                rewards_per_episode, actor_losses, critic_losses, entropies,
                best_reward, start_episode, global_step, ep_reward_buf,
                last_graph_update_time,
            )
        else:
            self._eval_loop(
                envs, policy,
                best_eval_reward, reward_file, video_file,
            )

        if writer is not None:
            writer.close()
        envs.close()

    def _train_loop(self, envs, policy, optimizer, writer,
                    rewards_per_episode, actor_losses, critic_losses, entropies,
                    best_reward, start_episode, global_step, ep_reward_buf,
                    last_graph_update_time):

        obs_space = envs.single_observation_space
        obs, _ = envs.reset()
        train_start = time.time()

        for episode in itertools.count(start_episode):
            episode_start_time = time.time()

            # ── 1. Rollout ──
            all_obs = np.zeros((self.rollout_steps, self.num_envs, *obs_space.shape), dtype=np.uint8)
            all_actions = np.zeros((self.rollout_steps, self.num_envs),
                                   dtype=np.int64 if self.is_discrete else np.float32)
            all_logprobs = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
            all_values = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
            all_rewards = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
            all_dones = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)

            episodes_completed = 0

            for step in range(self.rollout_steps):
                all_obs[step] = obs
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                    dist, value = policy(obs_t)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    if not self.is_discrete:
                        logprob = logprob.sum(-1)

                all_actions[step] = action.cpu().numpy()
                all_logprobs[step] = logprob.cpu().numpy()
                all_values[step] = value.squeeze(-1).cpu().numpy()

                obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated).astype(np.float32)
                all_rewards[step] = reward
                all_dones[step] = done

                ep_reward_buf += reward
                for i, d in enumerate(done):
                    if d:
                        rewards_per_episode.append(float(ep_reward_buf[i]))
                        ep_reward_buf[i] = 0.0
                        episodes_completed += 1

                global_step += self.num_envs

            # Bootstrap
            with torch.no_grad():
                next_value = policy.get_value(
                    torch.tensor(obs, dtype=torch.float32, device=device)
                ).squeeze(-1).cpu().numpy()

            # ── 2. GAE ──
            advantages, returns = compute_gae(
                all_rewards, all_values, all_dones, next_value,
                self.gamma, self.gae_lambda,
            )

            b_obs = torch.tensor(all_obs.reshape(-1, *obs_space.shape), dtype=torch.float32, device=device)
            b_actions = torch.tensor(all_actions.reshape(-1), device=device)
            b_logprobs = torch.tensor(all_logprobs.reshape(-1), dtype=torch.float32, device=device)
            b_advantages = torch.tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
            b_returns = torch.tensor(returns.reshape(-1), dtype=torch.float32, device=device)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # ── 3. PPO update ──
            total_samples = b_obs.shape[0]
            actor_loss_accum = 0.0
            critic_loss_accum = 0.0
            entropy_accum = 0.0
            n_updates = 0

            for _ in range(self.update_epochs):
                indices = torch.randperm(total_samples)
                for start in range(0, total_samples, self.batch_size):
                    idx = indices[start: start + self.batch_size]

                    dist, new_values = policy(b_obs[idx])
                    new_logprobs = dist.log_prob(b_actions[idx])
                    if not self.is_discrete:
                        new_logprobs = new_logprobs.sum(-1)
                    entropy = dist.entropy()
                    if not self.is_discrete:
                        entropy = entropy.sum(-1)

                    ratio = (new_logprobs - b_logprobs[idx]).exp()
                    adv = b_advantages[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * adv

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.functional.mse_loss(new_values.squeeze(-1), b_returns[idx])
                    entropy_loss = -self.ent_coef * entropy.mean()
                    loss = actor_loss + self.vf_coef * critic_loss + entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                    optimizer.step()

                    actor_loss_accum += actor_loss.item()
                    critic_loss_accum += critic_loss.item()
                    entropy_accum += entropy.mean().item()
                    n_updates += 1

            al = actor_loss_accum / max(n_updates, 1)
            cl = critic_loss_accum / max(n_updates, 1)
            en = entropy_accum / max(n_updates, 1)
            actor_losses.append(al)
            critic_losses.append(cl)
            entropies.append(en)

            # ── 4. Logging ──
            mr = float(np.mean(rewards_per_episode[-100:])) if rewards_per_episode else 0.0
            sps = global_step / (time.time() - train_start) if time.time() > train_start else 0
            episode_elapsed = time.time() - episode_start_time

            # Use mean of last 100 episode rewards as the "episode reward" for save_model
            episode_reward = mr

            print(
                f"Episode {episode} | MeanReward100: {mr:0.1f} | "
                f"Actor: {al:.4f} | Critic: {cl:.4f} | Ent: {en:.3f} | "
                f"Steps: {global_step} | {sps:.0f} sps | "
                f"Episodes completed: {episodes_completed}"
            )

            if writer:
                writer.add_scalar("reward/mean_100", mr, global_step)
                writer.add_scalar("reward/best", best_reward if mr <= best_reward else mr, global_step)
                writer.add_scalar("losses/actor_loss", al, global_step)
                writer.add_scalar("losses/critic_loss", cl, global_step)
                writer.add_scalar("losses/entropy", en, global_step)
                writer.add_scalar("training/steps_per_second", sps, global_step)
                if rewards_per_episode:
                    writer.add_scalar("reward/episode", rewards_per_episode[-1], global_step)

            self.save_model(
                policy, optimizer, episode_reward, episode, best_reward,
                rewards_per_episode, actor_losses, critic_losses, entropies, global_step,
            )
            if episode_reward > best_reward:
                best_reward = episode_reward

            # Update graph every 10 seconds
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                save_graph(self.GRAPH_FILE, rewards_per_episode, actor_losses, critic_losses, entropies)
                last_graph_update_time = current_time

    def _eval_loop(self, env, policy, best_eval_reward, reward_file, video_file):
        for episode in itertools.count():
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0

            while not terminated and not truncated:
                obs_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    dist, _ = policy(obs_t)
                    if self.is_discrete:
                        action = dist.probs.argmax(dim=-1).item()
                    else:
                        action = dist.mean.squeeze(0).cpu().numpy()

                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                print(f"Episode Reward: {episode_reward:0.1f}, Step Reward: {reward:0.1f}", end="\r")

            print(f"\nEpisode {episode} | Reward: {episode_reward:.1f} | Best: {best_eval_reward:.1f}")

            if hasattr(env, "save_video") and episode_reward > best_eval_reward:
                best_eval_reward = episode_reward
                env.save_video(video_file)
                with open(reward_file, "w") as f:
                    f.write(f"{best_eval_reward}\n")
                print(f"  -> New best! Video saved to {video_file}")
