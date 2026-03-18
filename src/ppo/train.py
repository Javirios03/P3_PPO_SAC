import torch
import torch.nn as nn
from torch.distributions import Normal
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
        mu_logstd = self.actor(x)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        mu = torch.tanh(mu)  # [-1, 1] para acciones MuJoCo
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        
        # Critic
        value = self.critic(x)
        return dist, value, log_std

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
        self.envs = make_vec_env(self.config["env_id"], "pixel", self.config["num_envs"])
        self.obs_space = self.envs.observation_space
        self.action_space = self.envs.action_space
        
        self.policy = ActorCritic(self.obs_space, self.action_space).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config["learning_rate"])
        
        self.total_timesteps = self.config["total_timesteps"]
        self.rollout_steps = self.config["rollout_steps"]
        
    def run(self, is_training=True):
        obs, _ = self.envs.reset()
        for global_step in range(self.total_timesteps // self.rollout_steps):
            # Recoger rollout
            actions = []
            logprobs = []
            rewards = []
            values = []
            dones = []
            
            for step in range(self.rollout_steps):
                with torch.no_grad():
                    dist, value, _ = self.policy(torch.tensor(obs).to(device))
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    
                actions.append(action.cpu().numpy())
                logprobs.append(logp.cpu().numpy())
                values.append(value.squeeze().cpu().numpy())
                
                obs, reward, done, _, _ = self.envs.step(action.cpu().numpy())
                rewards.append(reward)
                dones.append(done)
            
            # GAE
            advantages = compute_gae(rewards, values, dones, 
                                   self.config["gamma"], self.config["gae_lambda"])
            
            # Normalizar advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update (simplificado)
            if is_training:
                for epoch in range(4):  # self.config["update_epochs"]
                    for batch_start in range(0, len(actions), 32):
                        batch_end = batch_start + 32
                        batch_actions = torch.tensor(actions[batch_start:batch_end]).to(device)
                        batch_logprobs = torch.tensor(logprobs[batch_start:batch_end]).to(device)
                        batch_values = torch.tensor(values[batch_start:batch_end]).to(device)
                        batch_advantages = torch.tensor(advantages[batch_start:batch_end]).to(device)
                        
                        dist, new_values, _ = self.policy(torch.tensor(obs[batch_start:batch_end]).to(device))
                        new_logprobs = dist.log_prob(batch_actions)
                        entropy = dist.entropy()
                        
                        ratio = torch.exp(new_logprobs - batch_logprobs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(new_values, batch_values)
                        entropy_loss = -0.01 * entropy.mean()
                        
                        loss = actor_loss + 0.5 * critic_loss + entropy_loss
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                        self.optimizer.step()
            
            obs = obs  # ya actualizado
        
        self.envs.close()
