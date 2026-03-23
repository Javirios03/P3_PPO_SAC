import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, device, RUNS_DIR
from src.wrappers import make_vec_env, make_env

# TensorBoard — importación opcional: si no está instalado, el logging se omite
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    print("[PPO] TensorBoard no disponible. Instala tensorboard para activarlo.")


# ---------------------------------------------------------------------------
# Utilidad: calcular la forma de salida de la CNN dado el obs_space
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
# Red Actor-Critic compartida
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        obs_shape = obs_space.shape  # (4, 84, 84)

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
            mu  = self.actor_mu(x)
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
    last_gae   = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_val = values[t + 1]
        delta    = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


# ---------------------------------------------------------------------------
# Utilidades de logging / gráficas
# ---------------------------------------------------------------------------
def _save_plots(run_dir, all_ep_rewards, actor_losses, critic_losses, entropies, timesteps):
    """Genera rewards.png y losses.png en run_dir/plots/."""
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Recompensa
    fig, ax = plt.subplots(figsize=(9, 4))
    rewards_arr = np.array(all_ep_rewards)
    mean_r = np.array([
        rewards_arr[max(0, i - 99): i + 1].mean()
        for i in range(len(rewards_arr))
    ])
    ax.plot(rewards_arr, alpha=0.25, color="steelblue", label="ep reward")
    ax.plot(mean_r, color="steelblue", linewidth=2, label="media(100 eps)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa")
    ax.set_title("Evolución de la recompensa (entrenamiento)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "rewards.png"), dpi=120)
    plt.close(fig)

    # Pérdidas
    if actor_losses:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(timesteps, actor_losses,  color="tomato")
        axes[0].set_title("Actor loss")
        axes[0].set_xlabel("Timestep")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(timesteps, critic_losses, color="seagreen")
        axes[1].set_title("Critic loss")
        axes[1].set_xlabel("Timestep")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(timesteps, entropies, color="mediumpurple")
        axes[2].set_title("Entropía media")
        axes[2].set_xlabel("Timestep")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "losses.png"), dpi=120)
        plt.close(fig)

    print(f"[PPO] Gráficas guardadas en {plots_dir}/")


# ---------------------------------------------------------------------------
# Agente PPO
# ---------------------------------------------------------------------------
class PPOAgent:
    def __init__(self, config_name):
        self.config      = load_config(config_name)
        self.config_name = config_name
        print(f"[PPO] Config: {self.config}")
        print(f"[PPO] Device: {device}")

        self.envs         = make_vec_env(self.config["env_id"], "pixel", self.config["num_envs"])
        self.obs_space    = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space
        print(f"[PPO] Obs space:    {self.obs_space}")
        print(f"[PPO] Action space: {self.action_space}")

        self.policy = ActorCritic(self.obs_space, self.action_space).to(device)
        print(f"[PPO] Parámetros: {sum(p.numel() for p in self.policy.parameters()):,}")

        lr = float(self.config["learning_rate"])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self.total_timesteps = int(self.config["total_timesteps"])
        self.rollout_steps   = int(self.config["rollout_steps"])
        self.num_envs        = int(self.config["num_envs"])
        self.batch_size      = int(self.config["batch_size"])
        self.update_epochs   = int(self.config["update_epochs"])

        self.run_dir = os.path.join(RUNS_DIR, config_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.log_path = os.path.join(self.run_dir, "training_log.csv")

        # TensorBoard writer
        self.writer = None
        if _TB_AVAILABLE:
            tb_dir = os.path.join(self.run_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir=tb_dir)
            print(f"[PPO] TensorBoard activo → tensorboard --logdir {tb_dir}")

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    def _train(self):
        obs, _ = self.envs.reset()

        total_updates  = self.total_timesteps // (self.rollout_steps * self.num_envs)
        global_step    = 0
        ep_rewards_all = []
        ep_reward_buf  = np.zeros(self.num_envs)

        plot_timesteps   = []
        plot_actor_loss  = []
        plot_critic_loss = []
        plot_entropy     = []

        with open(self.log_path, "w") as f:
            f.write("update,timestep,mean_ep_reward,actor_loss,critic_loss,entropy,sps\n")

        start_time = time.time()

        for update in range(1, total_updates + 1):
            # ── 1. Rollout ────────────────────────────────────────────
            all_obs      = np.zeros((self.rollout_steps, self.num_envs, *self.obs_space.shape), dtype=np.uint8)
            all_actions  = np.zeros((self.rollout_steps, self.num_envs),
                                    dtype=np.int64 if self.is_discrete else np.float32)
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
                    if not self.is_discrete:
                        logprob = logprob.sum(-1)

                all_actions[step]  = action.cpu().numpy()
                all_logprobs[step] = logprob.cpu().numpy()
                all_values[step]   = value.squeeze(-1).cpu().numpy()

                obs, reward, terminated, truncated, _ = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated).astype(np.float32)
                all_rewards[step] = reward
                all_dones[step]   = done

                ep_reward_buf += reward
                for i, d in enumerate(done):
                    if d:
                        ep_rewards_all.append(float(ep_reward_buf[i]))
                        ep_reward_buf[i] = 0.0

                global_step += self.num_envs

            # Bootstrap
            with torch.no_grad():
                next_value = self.policy.get_value(
                    torch.tensor(obs, dtype=torch.float32).to(device)
                ).squeeze(-1).cpu().numpy()

            # ── 2. GAE ────────────────────────────────────────────────
            advantages, returns = compute_gae(
                all_rewards, all_values, all_dones, next_value,
                float(self.config["gamma"]), float(self.config["gae_lambda"])
            )

            b_obs        = torch.tensor(all_obs.reshape(-1, *self.obs_space.shape), dtype=torch.float32).to(device)
            b_actions    = torch.tensor(all_actions.reshape(-1)).to(device)
            b_logprobs   = torch.tensor(all_logprobs.reshape(-1), dtype=torch.float32).to(device)
            b_advantages = torch.tensor(advantages.reshape(-1), dtype=torch.float32).to(device)
            b_returns    = torch.tensor(returns.reshape(-1), dtype=torch.float32).to(device)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # ── 3. Actualización PPO ──────────────────────────────────
            total_samples     = b_obs.shape[0]
            actor_loss_accum  = 0.0
            critic_loss_accum = 0.0
            entropy_accum     = 0.0
            n_updates         = 0

            for _ in range(self.update_epochs):
                indices = torch.randperm(total_samples)
                for start in range(0, total_samples, self.batch_size):
                    idx = indices[start: start + self.batch_size]

                    dist, new_values = self.policy(b_obs[idx])
                    new_logprobs = dist.log_prob(b_actions[idx])
                    if not self.is_discrete:
                        new_logprobs = new_logprobs.sum(-1)
                    entropy = dist.entropy()
                    if not self.is_discrete:
                        entropy = entropy.sum(-1)

                    ratio = (new_logprobs - b_logprobs[idx]).exp()
                    adv   = b_advantages[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio,
                                        1 - float(self.config["clip_coef"]),
                                        1 + float(self.config["clip_coef"])) * adv

                    actor_loss   = -torch.min(surr1, surr2).mean()
                    critic_loss  = nn.functional.mse_loss(new_values.squeeze(-1), b_returns[idx])
                    entropy_loss = -float(self.config["ent_coef"]) * entropy.mean()
                    loss = actor_loss + float(self.config["vf_coef"]) * critic_loss + entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.config["max_grad_norm"]))
                    self.optimizer.step()

                    actor_loss_accum  += actor_loss.item()
                    critic_loss_accum += critic_loss.item()
                    entropy_accum     += entropy.mean().item()
                    n_updates         += 1

            # ── 4. Logging ────────────────────────────────────────────
            al  = actor_loss_accum  / max(n_updates, 1)
            cl  = critic_loss_accum / max(n_updates, 1)
            en  = entropy_accum     / max(n_updates, 1)
            mr  = float(np.mean(ep_rewards_all[-100:])) if ep_rewards_all else 0.0
            sps = global_step / (time.time() - start_time)

            plot_timesteps.append(global_step)
            plot_actor_loss.append(al)
            plot_critic_loss.append(cl)
            plot_entropy.append(en)

            if update % 5 == 0 or update == 1:
                print(
                    f"Update {update:4d}/{total_updates} | step {global_step:8d} | "
                    f"reward {mr:7.2f} | actor {al:.4f} | critic {cl:.4f} | "
                    f"ent {en:.3f} | {sps:.0f} sps"
                )

            with open(self.log_path, "a") as f:
                f.write(f"{update},{global_step},{mr:.4f},{al:.6f},{cl:.6f},{en:.6f},{sps:.1f}\n")

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("losses/actor_loss",  al,  global_step)
                self.writer.add_scalar("losses/critic_loss", cl,  global_step)
                self.writer.add_scalar("losses/entropy",     en,  global_step)
                self.writer.add_scalar("charts/mean_reward_100ep", mr, global_step)
                self.writer.add_scalar("charts/sps",         sps, global_step)
                if ep_rewards_all:
                    self.writer.add_scalar("charts/episode_reward",
                                           ep_rewards_all[-1], global_step)

            # ── 5. Checkpoint cada 50 updates ─────────────────────────
            if update % 50 == 0:
                self._save_checkpoint(update, global_step)

        # Final
        self._save_checkpoint("final", global_step)
        _save_plots(self.run_dir, ep_rewards_all,
                    plot_actor_loss, plot_critic_loss, plot_entropy,
                    plot_timesteps)
        if self.writer:
            self.writer.close()

        self.envs.close()
        print(f"[PPO] Entrenamiento completado. Pasos totales: {global_step:,}")

    # ------------------------------------------------------------------
    # EVALUACIÓN
    # ------------------------------------------------------------------
    def _eval(self, n_episodes=10, render=True):
        """
        Carga el checkpoint final y ejecuta n_episodes con política determinista.
        """
        ckpt_path = os.path.join(self.run_dir, "checkpoint_final.pt")
        if not os.path.exists(ckpt_path):
            # Buscar el checkpoint numerado más reciente
            ckpts = sorted([
                f for f in os.listdir(self.run_dir)
                if f.startswith("checkpoint_") and f.endswith(".pt")
            ])
            if not ckpts:
                print("[PPO] No se encontró ningún checkpoint. Entrena primero con --train.")
                self.envs.close()
                return
            ckpt_path = os.path.join(self.run_dir, ckpts[-1])

        print(f"[PPO] Cargando checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()
        print(f"[PPO] Update {ckpt.get('update', '?')} | "
              f"step {ckpt.get('global_step', 0):,}")

        # Entorno individual con render
        self.envs.close()   # cerramos el vec env que ya no necesitamos
        eval_env = make_env(self.config["env_id"], obs_type="pixel",
                            render=render, seed=999)

        ep_rewards = []
        ep_lengths = []

        for ep in range(n_episodes):
            obs, _ = eval_env.reset()
            ep_reward = 0.0
            ep_len    = 0
            done      = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    dist, _ = self.policy(obs_t)
                    # Acción determinista
                    if self.is_discrete:
                        action = dist.probs.argmax(dim=-1).item()
                    else:
                        action = dist.mean.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done       = terminated or truncated
                ep_reward += reward
                ep_len    += 1

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)
            print(f"  Ep {ep + 1:2d}/{n_episodes} | reward: {ep_reward:7.2f} | length: {ep_len}")

        eval_env.close()
        print(f"\n[PPO] Media: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f} | "
              f"Length media: {np.mean(ep_lengths):.1f}")

    # ------------------------------------------------------------------
    # PUNTO DE ENTRADA PÚBLICO
    # ------------------------------------------------------------------
    def run(self, is_training=True, render=True):
        """
        --train  → is_training=True  → entrena y guarda checkpoints + gráficas.
        (nada)   → is_training=False → carga checkpoint y evalúa con render.
        """
        # Propiedad de conveniencia para no repetir isinstance en todo el código
        self.is_discrete = isinstance(self.action_space, gym.spaces.Discrete)

        if is_training:
            self._train()
        else:
            self._eval(n_episodes=10, render=render)

    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag, global_step):
        path = os.path.join(self.run_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "update":      tag,
            "global_step": global_step,
            "policy":      self.policy.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
        }, path)
        print(f"  → Checkpoint: {path}")