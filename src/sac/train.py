import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter

from src.utils import load_config, device, RUNS_DIR, DATE_FORMAT, save_graph
from src.wrappers import make_env

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Conv encoder (compartido entre actor y critics)
# ---------------------------------------------------------------------------
def _build_encoder(obs_shape):
    """Devuelve (encoder_module, conv_out_dim)."""
    net = nn.Sequential(
        nn.Conv2d(obs_shape[0], 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        dummy = torch.zeros(1, *obs_shape)
        out_dim = net(dummy).shape[1]
    return net, out_dim


# ---------------------------------------------------------------------------
# Actor  (política Categorical soft — válido para espacios Discrete)
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.encoder, conv_out = _build_encoder(obs_shape)
        self.head = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)

    def forward(self, obs):
        """Devuelve (action_sampled, log_prob, probs, log_probs_all)."""
        x = obs / 255.0
        logits = self.head(self.encoder(x))
        log_probs_all = F.log_softmax(logits, dim=-1)
        probs = log_probs_all.exp()
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = log_probs_all.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, log_prob, probs, log_probs_all

    def get_action(self, obs):
        return self.forward(obs)


# ---------------------------------------------------------------------------
# Twin Critic  Q1 y Q2 en un solo módulo (evita duplicar el encoder)
# ---------------------------------------------------------------------------
class TwinCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        # Dos encoders independientes (importante para estabilidad)
        self.encoder1, conv_out = _build_encoder(obs_shape)
        self.encoder2, _ = _build_encoder(obs_shape)
        self.q1 = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        """Devuelve (Q1_all_actions, Q2_all_actions) — shape (B, n_actions)."""
        x = obs / 255.0
        return self.q1(self.encoder1(x)), self.q2(self.encoder2(x))


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos]      = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.dones[self.pos]    = done
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx],      dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[idx],  dtype=torch.long,    device=self.device),
            torch.tensor(self.rewards[idx],  dtype=torch.float32, device=self.device),
            torch.tensor(self.next_obs[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[idx],    dtype=torch.float32, device=self.device),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------
class SACAgent:

    def __init__(self, hyperparameter_set):
        config = load_config(hyperparameter_set)
        self.config = config
        self.hyperparameter_set = hyperparameter_set

        self.env_id          = config["env_id"]
        self.obs_type        = config.get("obs", "pixel")
        self.buffer_size     = int(config.get("buffer_size", 100_000))
        self.batch_size      = int(config.get("batch_size", 256))
        self.learning_rate   = float(config.get("learning_rate", 3e-4))
        self.gamma           = float(config.get("gamma", 0.99))
        self.tau             = float(config.get("tau", 0.005))
        self.alpha           = float(config.get("alpha", 0.2))
        self.auto_alpha      = bool(config.get("auto_alpha", True))
        self.learning_starts = int(config.get("learning_starts", 1000))
        self.train_freq      = int(config.get("train_freq", 1))
        self.max_grad_norm   = float(config.get("max_grad_norm", 10.0))
        self.target_entropy_coef = float(config.get("target_entropy_coef", 0.50))
        self.alpha_lr        = float(config.get("alpha_lr", 1e-4))
        self.env_kwargs      = config.get("env_kwargs", {})

        self.LOG_FILE        = os.path.join(RUNS_DIR, hyperparameter_set, "training.log")
        self.MODEL_FILE      = os.path.join(RUNS_DIR, hyperparameter_set, "best_model.pt")
        self.CHECKPOINT_FILE = os.path.join(RUNS_DIR, hyperparameter_set, "checkpoint.pt")
        self.GRAPH_FILE      = os.path.join(RUNS_DIR, hyperparameter_set, "graph.png")
        self.TB_DIR          = os.path.join(RUNS_DIR, hyperparameter_set, "tensorboard")

    # ------------------------------------------------------------------
    # Construcción de redes y estado de entrenamiento
    # ------------------------------------------------------------------
    def _build_networks(self, obs_space, action_space):
        obs_shape = obs_space.shape
        n_actions = action_space.n

        actor  = Actor(obs_shape, n_actions).to(device)
        critic = TwinCritic(obs_shape, n_actions).to(device)

        critic_target = TwinCritic(obs_shape, n_actions).to(device)
        critic_target.load_state_dict(critic.state_dict())
        for p in critic_target.parameters():
            p.requires_grad = False

        actor_opt  = torch.optim.Adam(actor.parameters(),  lr=self.learning_rate)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=self.learning_rate)

        log_alpha = None
        alpha_opt = None
        if self.auto_alpha:
            self.target_entropy = -np.log(1.0 / n_actions) * self.target_entropy_coef
            log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32,
                                    device=device, requires_grad=True)
            alpha_opt = torch.optim.Adam([log_alpha], lr=self.alpha_lr)

        print(f"[SAC] Actor params:  {sum(p.numel() for p in actor.parameters()):,}")
        print(f"[SAC] Critic params: {sum(p.numel() for p in critic.parameters()):,}")
        return actor, critic, critic_target, actor_opt, critic_opt, log_alpha, alpha_opt

    # ------------------------------------------------------------------
    # Soft update del target critic
    # ------------------------------------------------------------------
    @staticmethod
    def _soft_update(net, target, tau):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

    # ------------------------------------------------------------------
    # Un paso de actualización SAC
    # ------------------------------------------------------------------
    def _update(self, batch, actor, critic, critic_target,
            actor_opt, critic_opt, log_alpha, alpha_opt):

        obs, actions, rewards, next_obs, dones = batch
        alpha = log_alpha.exp().item() if self.auto_alpha else self.alpha

        with torch.no_grad():
            _, _, next_probs, next_log_probs_all = actor.get_action(next_obs)
            q1_next, q2_next = critic_target(next_obs)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (min_q_next - alpha * next_log_probs_all)).sum(dim=1)
            target_q = rewards + self.gamma * (1.0 - dones) * v_next

        # ── Critic loss ──
        q1_all, q2_all = critic(obs)
        q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        # critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        critic_loss = 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)

        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
        critic_opt.step()

        # ── Actor loss ──
        # FIX 2: pasada fresca del critic con detach — no reutilizar tensores del paso anterior
        _, _, probs, log_probs_all = actor.get_action(obs)
        q1_pi, q2_pi = critic(obs)
        min_q_pi = torch.min(q1_pi, q2_pi).detach()  # detach: no queremos gradientes del critic aquí
        actor_loss = (probs * (alpha * log_probs_all - min_q_pi)).sum(dim=1).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
        actor_opt.step()

        # ── Alpha loss ──
        alpha_loss = 0.0
        if self.auto_alpha:
            with torch.no_grad():
                entropy = -(probs * log_probs_all).sum(dim=1).mean()
            alpha_loss_t = log_alpha.exp() * (entropy - self.target_entropy)
            alpha_opt.zero_grad()
            alpha_loss_t.backward()
            alpha_opt.step()
            alpha_loss = alpha_loss_t.item()
            with torch.no_grad():
                log_alpha.clamp_(min=np.log(0.01), max=np.log(5.0))

        self._soft_update(critic, critic_target, self.tau)

        return actor_loss.item(), critic_loss.item(), alpha_loss, alpha

    # ------------------------------------------------------------------
    # Checkpoint / save
    # ------------------------------------------------------------------
    def save_model(self, actor, critic, actor_opt, critic_opt, log_alpha, alpha_opt,
                   episode_reward, episode, best_reward,
                   rewards_per_episode, actor_losses, critic_losses, entropies, global_step):

        if episode_reward > best_reward:
            msg = (f"{datetime.now().strftime(DATE_FORMAT)}: New best reward "
                   f"{episode_reward:.1f} at episode {episode}, saving model...")
            print(msg)
            with open(self.LOG_FILE, "a") as f:
                f.write(msg + "\n")
            torch.save(actor.state_dict(), self.MODEL_FILE)

        elif episode % 100 == 0:
            ckpt = {
                "actor_state_dict":    actor.state_dict(),
                "critic_state_dict":   critic.state_dict(),
                "actor_opt":           actor_opt.state_dict(),
                "critic_opt":          critic_opt.state_dict(),
                "log_alpha":           log_alpha.item() if log_alpha is not None else None,
                "alpha_opt":           alpha_opt.state_dict() if alpha_opt is not None else None,
                "episode":             episode,
                "best_reward":         best_reward,
                "rewards_per_episode": rewards_per_episode,
                "actor_losses":        actor_losses,
                "critic_losses":       critic_losses,
                "entropies":           entropies,
                "global_step":         global_step,
            }
            if os.path.exists(self.CHECKPOINT_FILE):
                os.remove(self.CHECKPOINT_FILE)
            torch.save(ckpt, self.CHECKPOINT_FILE)
            msg = f"{datetime.now().strftime(DATE_FORMAT)}: Checkpoint at episode {episode}"
            print(msg)
            with open(self.LOG_FILE, "a") as f:
                f.write(msg + "\n")

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------
    def run(self, is_training=True, render=False):
        writer = None
        if is_training:
            start_time = datetime.now()
            last_graph_time = start_time
            msg = f"{start_time.strftime(DATE_FORMAT)}: SAC Training starting..."
            print(msg)
            with open(self.LOG_FILE, "a") as f:
                f.write(msg + "\n")
            writer = SummaryWriter(log_dir=self.TB_DIR)

        env = make_env(self.env_id, self.obs_type, render=render,
                       seed=42, env_kwargs=self.env_kwargs)
        obs_space    = env.observation_space
        action_space = env.action_space

        actor, critic, critic_target, actor_opt, critic_opt, log_alpha, alpha_opt = \
            self._build_networks(obs_space, action_space)

        # ── Estado persistente ──
        start_episode    = 0
        best_reward      = float("-inf")
        best_eval_reward = float("-inf")
        global_step      = 0
        rewards_per_episode = []
        actor_losses, critic_losses, entropies = [], [], []

        replay_buffer = ReplayBuffer(self.buffer_size, obs_space.shape, device=device)

        # ── Resume ──
        if is_training and os.path.exists(self.CHECKPOINT_FILE):
            ckpt = torch.load(self.CHECKPOINT_FILE, map_location=device, weights_only=False)
            actor.load_state_dict(ckpt["actor_state_dict"])
            critic.load_state_dict(ckpt["critic_state_dict"])
            critic_target.load_state_dict(ckpt["critic_state_dict"])
            actor_opt.load_state_dict(ckpt["actor_opt"])
            critic_opt.load_state_dict(ckpt["critic_opt"])
            if self.auto_alpha and ckpt.get("log_alpha") is not None:
                with torch.no_grad():
                    log_alpha.fill_(ckpt["log_alpha"])
                alpha_opt.load_state_dict(ckpt["alpha_opt"])
            start_episode       = int(ckpt.get("episode", 0)) + 1
            best_reward         = float(ckpt.get("best_reward", best_reward))
            rewards_per_episode = list(ckpt.get("rewards_per_episode", []))
            actor_losses        = list(ckpt.get("actor_losses", []))
            critic_losses       = list(ckpt.get("critic_losses", []))
            entropies           = list(ckpt.get("entropies", []))
            global_step         = int(ckpt.get("global_step", 0))
            print(f"Resumed from episode {start_episode - 1} | global_step={global_step}")

        elif not is_training and os.path.exists(self.MODEL_FILE):
            actor.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            actor.eval()
            print(f"Loaded actor weights from: {self.MODEL_FILE}")

        # Eval mode
        if not is_training:
            eval_dir    = os.path.join(RUNS_DIR, self.hyperparameter_set, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            reward_file = os.path.join(eval_dir, "best_reward.txt")
            video_file  = os.path.join(eval_dir, "best.mp4")
            if os.path.exists(reward_file):
                with open(reward_file) as f:
                    best_eval_reward = float(f.read().strip())
            self._eval_loop(env, actor, best_eval_reward, reward_file, video_file)
            env.close()
            return

        # ── Training loop ──
        train_start = time.time()

        for episode in itertools.count(start_episode):
            obs, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0
            ep_actor_losses, ep_critic_losses, ep_entropies = [], [], []

            while not terminated and not truncated:
                # Acción: aleatoria hasta learning_starts, luego política
                if global_step < self.learning_starts:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        action, _, _, _ = actor.get_action(obs_t)
                        action = action.item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = float(terminated)   # truncated NO es terminal real
                replay_buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs
                episode_reward += reward
                global_step += 1

                # Update
                if global_step >= self.learning_starts and global_step % self.train_freq == 0:
                    batch = replay_buffer.sample(self.batch_size)
                    al, cl, _, current_alpha = self._update(
                        batch, actor, critic, critic_target,
                        actor_opt, critic_opt, log_alpha, alpha_opt,
                    )
                    ep_actor_losses.append(al)
                    ep_critic_losses.append(cl)
                    # Entropía aproximada de la política actual
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        _, _, probs, log_probs_all = actor.get_action(obs_t)
                        entropy = -(probs * log_probs_all).sum(dim=1).mean().item()
                    ep_entropies.append(entropy)

            rewards_per_episode.append(episode_reward)
            al  = float(np.mean(ep_actor_losses))  if ep_actor_losses  else 0.0
            cl  = float(np.mean(ep_critic_losses)) if ep_critic_losses else 0.0
            en  = float(np.mean(ep_entropies))     if ep_entropies     else 0.0
            actor_losses.append(al)
            critic_losses.append(cl)
            entropies.append(en)

            mr  = float(np.mean(rewards_per_episode[-100:]))
            sps = global_step / (time.time() - train_start + 1e-9)
            alpha_val = log_alpha.exp().item() if self.auto_alpha else self.alpha

            print(
                f"Episode {episode} | Reward: {episode_reward:.1f} | MeanR100: {mr:.1f} | "
                f"Actor: {al:.4f} | Critic: {cl:.4f} | Ent: {en:.3f} | "
                f"Alpha: {alpha_val:.4f} | Steps: {global_step} | {sps:.0f} sps"
            )

            if writer:
                writer.add_scalar("reward/episode",    episode_reward, global_step)
                writer.add_scalar("reward/mean_100",   mr,             global_step)
                writer.add_scalar("losses/actor_loss", al,             global_step)
                writer.add_scalar("losses/critic_loss",cl,             global_step)
                writer.add_scalar("losses/entropy",    en,             global_step)
                writer.add_scalar("sac/alpha",         alpha_val,      global_step)
                writer.add_scalar("training/steps_per_second", sps,    global_step)

            self.save_model(
                actor, critic, actor_opt, critic_opt, log_alpha, alpha_opt,
                mr, episode, best_reward,
                rewards_per_episode, actor_losses, critic_losses, entropies, global_step,
            )
            if mr > best_reward:
                best_reward = mr

            current_time = datetime.now()
            if current_time - last_graph_time > timedelta(seconds=10):
                save_graph(self.GRAPH_FILE, rewards_per_episode, actor_losses, critic_losses, entropies)
                last_graph_time = current_time

        if writer:
            writer.close()
        env.close()

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    def _eval_loop(self, env, actor, best_eval_reward, reward_file, video_file):
        for episode in itertools.count():
            obs, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0

            while not terminated and not truncated:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    _, _, probs, _ = actor.get_action(obs_t)
                    action = probs.argmax(dim=-1).item()   # acción greedy en eval

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                print(f"Episode Reward: {episode_reward:.1f}", end="\r")

            print(f"\nEpisode {episode} | Reward: {episode_reward:.1f} | Best: {best_eval_reward:.1f}")

            if hasattr(env, "save_video") and episode_reward > best_eval_reward:
                best_eval_reward = episode_reward
                env.save_video(video_file)
                with open(reward_file, "w") as f:
                    f.write(f"{best_eval_reward}\n")
                print(f"  -> New best! Video saved to {video_file}")