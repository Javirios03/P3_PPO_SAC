import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from gymnasium.wrappers import (
    AddRenderObservation,
    ResizeObservation,
    FrameStackObservation,
    HumanRendering,
)
import itertools


# Discreize continuous actions n into bins, no need to separete into combinations of actions for each dimension, as the agent will just choose one action at a time
# Example: For HalfCheetah-v5, action space is Box(-1.0, 1.0, (6,), float32)
# With bins=3, we create discrete actions for each dimension: [-1.0, 0.0, 1.0]
# The total number of discrete actions becomes 3x6 = 18
# but the action 0 is the same for all dimensions so we need to remove duplicates
# The resulting action space is Discrete(13) with actions
class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=3):
        super().__init__(env)

        # Nos aseguramos de que el entorno original sea continuo
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "Action space must be continuous (Box)."

        low = self.env.action_space.low
        high = self.env.action_space.high
        n_dims = self.env.action_space.shape[0]

        # 1. Empezamos la lista con la acción base: "No hacer nada" (Vector de ceros)
        actions = [np.zeros(n_dims, dtype=np.float32)]

        # 2. Iteramos por cada dimensión para crear sus acciones individuales
        for i in range(n_dims):
            # Generamos los valores posibles para esta articulación/motor
            values = np.linspace(low[i], high[i], bins)

            for v in values:
                # Evitamos añadir el 0.0 de nuevo, ya que está cubierto por la acción base
                if not np.isclose(v, 0.0):
                    # Creamos un vector de ceros y solo modificamos la dimensión actual
                    action_vec = np.zeros(n_dims, dtype=np.float32)
                    action_vec[i] = v
                    actions.append(action_vec)

        # Convertimos a array de numpy para acceso rápido en el step
        self.actions_grid = np.array(actions, dtype=np.float32)

        # 3. Definimos el nuevo espacio de acción discreto
        # Para bins=3 y n_dims=6, esto será Discrete(13)
        self.action_space = Discrete(len(self.actions_grid))
        print(
            f"DiscretizedActionWrapper initialized with {len(self.actions_grid)} discrete actions."
        )

    def action(self, action_index):
        # Mapea el entero que devuelve la DQN al vector continuo para MuJoCo
        idx = int(action_index.item() if hasattr(action_index, "item") else action_index)
        return self.actions_grid[idx]


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert RGB observations to grayscale using cv2 (faster than numpy)."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape  # (H, W, 3)
        self.observation_space = Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1]), dtype=np.uint8
        )

    def observation(self, obs):
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


class RenderGrayscaleWrapper(gym.ObservationWrapper):
    """Replace obs with grayscale render in a single wrapper.
    Expects env created with render_mode='rgb_array' and width/height
    already set to target resolution so no resize is needed."""

    def __init__(self, env, obs_size=84):
        super().__init__(env)
        self._obs_size = obs_size
        self.observation_space = Box(
            low=0, high=255, shape=(obs_size, obs_size), dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render()  # already at target resolution
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


class OpenCVRenderWrapper(gym.Wrapper):
    """Muestra el renderizado en una ventana flotante usando OpenCV."""

    def __init__(self, env, window_name="MuJoCo Preview"):
        super().__init__(env)
        self.window_name = window_name

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        img = self.env.render()
        if img is not None:
            # Convertir RGB a BGR para OpenCV y redimensionar para ver mejor
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.resize(img_bgr, (480, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(self.window_name, img_bgr)
            cv2.waitKey(1)
        return obs, reward, terminated, truncated, info

    def close(self):
        cv2.destroyAllWindows()
        return self.env.close()


class WalkerReward(gym.Wrapper):
    def __init__(self, env, torso_weight=1.0, knee_weight=0.3, symmetry_weight=0.1):
        super().__init__(env)
        self.torso_weight = torso_weight
        self.knee_weight = knee_weight
        self.symmetry_weight = symmetry_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        torso_angle = obs[1]
        left_knee = obs[3]
        right_knee = obs[6]
        left_hip = obs[2]
        right_hip = obs[5]

        torso_penalty = self.torso_weight * torso_angle**2
        knee_penalty = self.knee_weight * (left_knee**2 + right_knee**2)
        symmetry_penalty = self.symmetry_weight * (
            (left_knee - right_knee) ** 2 + (left_hip - right_hip) ** 2
        )

        # Shaped reward
        shaped_reward = reward - torso_penalty - knee_penalty - symmetry_penalty

        return obs, shaped_reward, terminated, truncated, info


def make_state_env(env_id, render=False, seed=42):

    if "Walker2d-v5" in env_id:
        env = gym.make(
            env_id,
            render_mode="human" if render else None,
            healthy_angle_range=(-0.4, 0.4),
            max_episode_steps=2500,
        )
        env = WalkerReward(env)
    else:
        env = gym.make(env_id, render_mode="human" if render else None)

    # Discretize actions if needed
    if isinstance(env.action_space, Box):
        env = DiscretizedActionWrapper(env, bins=3)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_pixel_env(env_id, render=False, seed=42):
    """
    Creates env with Pixel observation + Discretization + Stack.
    Renders directly at 84x84 to avoid expensive high-res render + resize.
    """
    obs_size = 84

    if "Walker2d-v5" in env_id:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            width=obs_size,
            height=obs_size,
            healthy_angle_range=(-0.4, 0.4),
            max_episode_steps=2500,
        )
        env = WalkerReward(env)
    else:
        env = gym.make(env_id, render_mode="rgb_array", width=obs_size, height=obs_size)

    # Single wrapper: render → grayscale (no resize needed, already 84x84)
    env = RenderGrayscaleWrapper(env, obs_size=obs_size)

    # Discretize actions only for continuous-control envs (MuJoCo Box).
    # For discrete-action envs (e.g., CartPole), keep the original Discrete action space.
    if isinstance(env.action_space, Box):
        env = DiscretizedActionWrapper(env, bins=3)

    # Stack frames
    env = FrameStackObservation(env, stack_size=4)

    if render:
        # env = HumanRendering(env)
        env = OpenCVRenderWrapper(env)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_env(env_id, obs_type, render=False, seed=42):
    if obs_type == "pixel":
        return make_pixel_env(env_id, render, seed)
    elif obs_type == "state":
        return make_state_env(env_id, render, seed)
    else:
        raise ValueError(f"Unsupported obs_type: {obs_type}")


def make_vec_env(env_id, obs_type, num_envs, seed=42):
    """Create a vectorized environment with num_envs parallel environments.
    Uses AsyncVectorEnv for pixel obs (parallel rendering in subprocesses)
    and SyncVectorEnv for state obs."""

    def _make_thunk(idx):
        def _thunk():
            return make_env(env_id, obs_type, render=False, seed=seed + idx)

        return _thunk

    thunks = [_make_thunk(i) for i in range(num_envs)]
    if obs_type == "pixel":
        # SyncVectorEnv for pixel: avoids per-subprocess OpenGL context
        # limits (MuJoCo framebuffer crash on Windows with many envs).
        return gym.vector.SyncVectorEnv(thunks)
    return gym.vector.SyncVectorEnv(thunks)
