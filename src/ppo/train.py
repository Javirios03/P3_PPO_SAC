import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import gymnasium as gym

from src.config import load_config, device, RUNS_DIR
from src.wrappers import make_vec_env


# ---------------------------------------------------------------------------
# Utilidad: calcular la forma de salida de la CNN dado el obs_space
# ---------------------------------------------------------------------------
def _conv_out_size(obs_shape):
    """
    Calcula el número de features que salen de la CNN estándar (3 conv layers)
    para una entrada de shape (C, H, W).
    """
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
# Red Actor-Critic compartida
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """
    Red Actor-Critic con CNN compartida para observaciones en píxeles.
    Soporta espacios de acción discretos (Categorical) y continuos (Normal).
    """
    def __init__(self, obs_space, action_space):
        super().__init__()

        # obs_space.shape puede ser (C, H, W) tras FrameStack en formato (stack, H, W)
        obs_shape = obs_space.shape  # e.g. (4, 84, 84)

        # CNN compartida
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
            # Salida: mu + log_std por separado
            self.actor_mu = nn.Sequential(
                nn.Linear(conv_out, 512), nn.ReLU(),
                nn.Linear(512, n_actions),
                nn.Tanh()   # acotamos mu a [-1,1] (escala después si hace falta)
            )
            # log_std como parámetro aprendible (independiente del estado)
            self.log_std = nn.Parameter(torch.zeros(n_actions))

        # Critic: V(s)
        self.critic = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Inicialización ortogonal (mejora estabilidad en PPO)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Cabeza actor con ganancia pequeña para exploración inicial
        if self.is_discrete:
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor_mu[-2].weight, gain=0.01)

    def forward(self, obs):
        """
        Devuelve (distribución, valor_estado).
        obs: tensor float32 [B, C, H, W] con valores 0-255.
        """
        x = obs / 255.0
        x = self.conv(x)

        if self.is_discrete:
            logits = self.actor(x)
            dist = Categorical(logits=logits)
        else:
            mu = self.actor_mu(x)
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)

        value = self.critic(x)
        return dist, value

    def get_value(self, obs):
        x = obs / 255.0
        x = self.conv(x)
        return self.critic(x)


# ---------------------------------------------------------------------------
# GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """
    rewards:    np.array [T, N]
    values:     np.array [T, N]  ← valores estimados en cada paso del rollout
    dones:      np.array [T, N]
    next_value: np.array [N]     ← V(s_{T}) bootstrap del último estado
    Devuelve advantages [T, N] y returns [T, N].
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value          # bootstrap del estado siguiente real
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Agente PPO
# ---------------------------------------------------------------------------
class PPOAgent:
    def __init__(self, config_name):
        self.config = load_config(config_name)
        self.config_name = config_name
        print(f"[PPO] Config: {self.config}")
        print(f"[PPO] Device: {device}")

        self.envs = make_vec_env(
            self.config["env_id"], "pixel", self.config["num_envs"]
        )
        # single_observation_space / single_action_space devuelven la forma
        # de UN solo entorno, sin la dimensión num_envs al frente.
        self.obs_space    = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

        print(f"[PPO] Obs space:    {self.obs_space}")
        print(f"[PPO] Action space: {self.action_space}")

        self.policy = ActorCritic(self.obs_space, self.action_space).to(device)
        print(f"[PPO] Parámetros: {sum(p.numel() for p in self.policy.parameters()):,}")

        lr = float(self.config["learning_rate"])
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, eps=1e-5
        )

        self.total_timesteps = int(self.config["total_timesteps"])
        self.rollout_steps   = int(self.config["rollout_steps"])
        self.num_envs        = int(self.config["num_envs"])
        self.batch_size      = int(self.config["batch_size"])
        self.update_epochs   = int(self.config["update_epochs"])

        # Directorio de runs para checkpoints y logs
        self.run_dir = os.path.join(RUNS_DIR, config_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Log CSV
        self.log_path = os.path.join(self.run_dir, "training_log.csv")
        with open(self.log_path, "w") as f:
            f.write("update,timestep,mean_reward,actor_loss,critic_loss,entropy\n")

    # ------------------------------------------------------------------
    def run(self, is_training=True):
        obs, _ = self.envs.reset()
        # obs shape: [N, C, H, W]

        total_updates = self.total_timesteps // (self.rollout_steps * self.num_envs)
        global_step   = 0
        episode_rewards = []   # buffer para calcular mean reward
        ep_reward_buf   = np.zeros(self.num_envs)

        start_time = time.time()

        for update in range(1, total_updates + 1):
            # ── 1. Recolección del rollout ────────────────────────────
            all_obs      = np.zeros((self.rollout_steps, self.num_envs, *self.obs_space.shape), dtype=np.uint8)
            all_actions  = np.zeros((self.rollout_steps, self.num_envs), dtype=np.int64 if isinstance(self.action_space, gym.spaces.Discrete) else np.float32)
            all_logprobs = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
            all_values   = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
            all_rewards  = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
            all_dones    = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)

            for step in range(self.rollout_steps):
                all_obs[step] = obs

                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                    dist, value = self.policy(obs_t)
                    action  = dist.sample()
                    logprob = dist.log_prob(action)
                    if not isinstance(self.action_space, gym.spaces.Discrete):
                        logprob = logprob.sum(-1)  # suma sobre dimensiones de acción

                all_actions[step]  = action.cpu().numpy()
                all_logprobs[step] = logprob.cpu().numpy()
                all_values[step]   = value.squeeze(-1).cpu().numpy()

                action_np = action.cpu().numpy()
                obs, reward, terminated, truncated, _ = self.envs.step(action_np)
                done = np.logical_or(terminated, truncated).astype(np.float32)

                all_rewards[step] = reward
                all_dones[step]   = done

                # Tracking de recompensas por episodio
                ep_reward_buf += reward
                for i, d in enumerate(done):
                    if d:
                        episode_rewards.append(ep_reward_buf[i])
                        ep_reward_buf[i] = 0.0

                global_step += self.num_envs

            # Bootstrap del valor del último estado
            with torch.no_grad():
                obs_t      = torch.tensor(obs, dtype=torch.float32).to(device)
                next_value = self.policy.get_value(obs_t).squeeze(-1).cpu().numpy()

            # ── 2. GAE ────────────────────────────────────────────────
            advantages, returns = compute_gae(
                all_rewards, all_values, all_dones, next_value,
                float(self.config["gamma"]), float(self.config["gae_lambda"])
            )

            # Flatten [T, N] → [T*N]
            b_obs      = torch.tensor(
                all_obs.reshape(-1, *self.obs_space.shape), dtype=torch.float32
            ).to(device)
            b_actions  = torch.tensor(all_actions.reshape(-1)).to(device)
            b_logprobs = torch.tensor(all_logprobs.reshape(-1), dtype=torch.float32).to(device)
            b_advantages = torch.tensor(advantages.reshape(-1), dtype=torch.float32).to(device)
            b_returns  = torch.tensor(returns.reshape(-1), dtype=torch.float32).to(device)

            # Normalización de ventajas
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # ── 3. Actualización PPO ──────────────────────────────────
            total_samples = b_obs.shape[0]
            actor_loss_epoch  = 0.0
            critic_loss_epoch = 0.0
            entropy_epoch     = 0.0
            n_updates         = 0

            if is_training:
                for epoch in range(self.update_epochs):
                    # SHUFFLE los datos en cada época (¡crucial para PPO!)
                    indices = torch.randperm(total_samples)

                    for start in range(0, total_samples, self.batch_size):
                        idx = indices[start:start + self.batch_size]

                        dist, new_values = self.policy(b_obs[idx])
                        new_logprobs = dist.log_prob(b_actions[idx])
                        if not isinstance(self.action_space, gym.spaces.Discrete):
                            new_logprobs = new_logprobs.sum(-1)
                        entropy = dist.entropy()
                        if not isinstance(self.action_space, gym.spaces.Discrete):
                            entropy = entropy.sum(-1)

                        # Ratio e = π_new / π_old (en log-space)
                        log_ratio = new_logprobs - b_logprobs[idx]
                        ratio     = log_ratio.exp()

                        adv   = b_advantages[idx]
                        surr1 = ratio * adv
                        surr2 = torch.clamp(
                            ratio,
                            1 - float(self.config["clip_coef"]),
                            1 + float(self.config["clip_coef"])
                        ) * adv

                        actor_loss  = -torch.min(surr1, surr2).mean()
                        # Critic loss con clipping opcional (mejora estabilidad)
                        v_pred      = new_values.squeeze(-1)
                        critic_loss = nn.functional.mse_loss(v_pred, b_returns[idx])
                        entropy_loss= -float(self.config["ent_coef"]) * entropy.mean()

                        loss = (actor_loss
                                + float(self.config["vf_coef"]) * critic_loss
                                + entropy_loss)

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            float(self.config["max_grad_norm"])
                        )
                        self.optimizer.step()

                        actor_loss_epoch  += actor_loss.item()
                        critic_loss_epoch += critic_loss.item()
                        entropy_epoch     += entropy.mean().item()
                        n_updates         += 1

            # ── 4. Logging ────────────────────────────────────────────
            mean_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            elapsed     = time.time() - start_time
            sps         = global_step / elapsed  # samples per second

            if update % 5 == 0 or update == 1:
                al = actor_loss_epoch  / max(n_updates, 1)
                cl = critic_loss_epoch / max(n_updates, 1)
                en = entropy_epoch     / max(n_updates, 1)
                print(
                    f"Update {update:4d}/{total_updates} | "
                    f"step {global_step:7d} | "
                    f"reward {mean_reward:7.2f} | "
                    f"actor {al:.4f} | critic {cl:.4f} | ent {en:.3f} | "
                    f"{sps:.0f} sps"
                )
                with open(self.log_path, "a") as f:
                    f.write(f"{update},{global_step},{mean_reward:.4f},{al:.6f},{cl:.6f},{en:.6f}\n")

            # ── 5. Checkpoint cada 50 updates ─────────────────────────
            if is_training and update % 50 == 0:
                ckpt = os.path.join(self.run_dir, f"checkpoint_{update}.pt")
                torch.save({
                    "update":       update,
                    "global_step":  global_step,
                    "policy":       self.policy.state_dict(),
                    "optimizer":    self.optimizer.state_dict(),
                }, ckpt)
                print(f"  → Checkpoint guardado: {ckpt}")

        # Checkpoint final
        if is_training:
            ckpt = os.path.join(self.run_dir, "checkpoint_final.pt")
            torch.save({
                "update":      total_updates,
                "global_step": global_step,
                "policy":      self.policy.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
            }, ckpt)
            print(f"[PPO] Entrenamiento completado. Checkpoint final: {ckpt}")

        self.envs.close()