import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np
from src.config import load_config, device
from src.wrappers import make_vec_env
from src.networks import Pixel_DQN  # reutilizamos CNN para píxeles

class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        # CNN compartida (tu Pixel_DQN sin las cabezas DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 7*7*64
        
        # Actor: gaussian policy sobre acciones continuas
        self.actor = nn.Sequential(
            nn.Linear(conv_out, 256), nn.ReLU(),
            nn.Linear(256, 2)  # mu (tanh) + log_std
        )
        
        # Critic: V(s)
        self.critic = nn.Sequential(
            nn.Linear(conv_out, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, obs):
        obs = obs / 255.0
        x = self.conv(obs)
        
        # Actor
        # mu_logstd = self.actor(x)
        # mu, log_std = mu_logstd.chunk(2, dim=-1)
        # mu = torch.tanh(mu)  # [-1, 1] para acciones MuJoCo
        # std = torch.exp(log_std)
        # dist = Normal(mu, std)
        logits = self.actor(x)          # [batch, 13]
        dist   = Categorical(logits=logits)
        
        # Critic
        value = self.critic(x)
        return dist, value

def compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_values = values[-1]
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
    return advantages

class PPOAgent:
    def __init__(self, config_name):
        self.config = load_config(config_name)
        # print(f"Loaded config: {list(self.config.keys())}")
        # print(f"Using device: {device}")
        # print(f"Full Config: {self.config}")
        self.envs = make_vec_env(self.config["env_id"], "pixel", self.config["num_envs"])
        self.obs_space = self.envs.observation_space
        self.action_space = self.envs.action_space
        
        self.policy = ActorCritic(self.obs_space, self.action_space).to(device)
        lr = float(self.config["learning_rate"])  # Delete once YAML parsing is fixed
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.total_timesteps = self.config["total_timesteps"]
        self.rollout_steps = self.config["rollout_steps"]
        
    def run(self, is_training=True):
        obs, _ = self.envs.reset()
        
        for global_step in range(self.total_timesteps // self.rollout_steps):
            # --- Rollout collection ---
            all_obs     = []
            all_actions = []
            all_logprobs= []
            all_rewards = []
            all_values  = []
            all_dones   = []

            for step in range(self.rollout_steps):
                all_obs.append(obs.copy())                          # save obs at this step

                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                    dist, value = self.policy(obs_t)
                    action   = dist.sample()
                    logp     = dist.log_prob(action)

                all_actions.append(action.cpu().numpy())
                all_logprobs.append(logp.cpu().numpy())
                all_values.append(value.squeeze(-1).cpu().numpy())

                obs, reward, done, _, _ = self.envs.step(action.cpu().numpy())
                all_rewards.append(reward)
                all_dones.append(done)

            # --- Flatten [T, N] → [T*N] ---
            all_obs_np      = np.array(all_obs)       # [T, N, C, H, W]
            all_actions_np  = np.array(all_actions)   # [T, N, action_dim]
            all_logprobs_np = np.array(all_logprobs)  # [T, N]
            all_values_np   = np.array(all_values)    # [T, N]
            all_rewards_np  = np.array(all_rewards)   # [T, N]
            all_dones_np    = np.array(all_dones)     # [T, N]

            b_obs      = torch.tensor(
                            all_obs_np.reshape(-1, *all_obs_np.shape[2:]),
                            dtype=torch.float32).to(device)                  # [T*N, C, H, W]
            b_actions  = torch.tensor(
                            all_actions_np.reshape(-1),
                            dtype=torch.long).to(device)                  # [T*N, action_dim]
            b_logprobs = torch.tensor(
                            all_logprobs_np.reshape(-1),
                            dtype=torch.float32).to(device)                  # [T*N]

            # GAE still operates on [T, N] — pass pre-flatten arrays
            advantages = compute_gae(
                all_rewards_np, all_values_np,
                all_dones_np, self.config["gamma"], self.config["gae_lambda"]
            )                                                                  # [T, N]

            b_advantages = torch.tensor(
                            advantages.reshape(-1),
                            dtype=torch.float32).to(device)                # [T*N]
            b_returns    = b_advantages + torch.tensor(
                            all_values_np.reshape(-1),
                            dtype=torch.float32).to(device)

            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # --- PPO update ---
            if is_training:
                total = b_obs.shape[0]
                for epoch in range(self.config["update_epochs"]):
                    for start in range(0, total, self.config["batch_size"]):
                        end = start + self.config["batch_size"]
                        dist, new_values = self.policy(b_obs[start:end])
                        new_logprobs = dist.log_prob(b_actions[start:end])
                        entropy = dist.entropy()

                        ratio  = torch.exp(new_logprobs - b_logprobs[start:end])
                        adv    = b_advantages[start:end]
                        surr1  = ratio * adv
                        surr2  = torch.clamp(ratio, 1 - self.config["clip_coef"],
                                                    1 + self.config["clip_coef"]) * adv

                        actor_loss  = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(new_values.squeeze(-1), b_returns[start:end])
                        entropy_loss= -self.config["ent_coef"] * entropy.mean()

                        loss = actor_loss + self.config["vf_coef"] * critic_loss + entropy_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["max_grad_norm"])
                        self.optimizer.step()

        self.envs.close()
