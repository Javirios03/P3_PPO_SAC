import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import (
    ResizeObservation,
    FrameStackObservation,
    AddRenderObservation,
)


# ---------------------------------------------------------------------------
# Discretización de acciones continuas
# ---------------------------------------------------------------------------
class DiscretizedActionWrapper(gym.ActionWrapper):
    """
    Convierte un espacio de acciones continuo Box en uno discreto.
    Acción 0 = vector de ceros ("no hacer nada").
    Para cada dimensión, añade bins-1 acciones (las que no son 0).
    Ejemplo: Walker2d (6 dims) con bins=3 → 1 + 6*2 = 13 acciones.
    """
    def __init__(self, env, bins=3):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), \
            "DiscretizedActionWrapper requiere espacio de acciones Box."

        low    = self.env.action_space.low
        high   = self.env.action_space.high
        n_dims = self.env.action_space.shape[0]

        actions = [np.zeros(n_dims, dtype=np.float32)]  # acción base: ceros
        for i in range(n_dims):
            values = np.linspace(low[i], high[i], bins)
            for v in values:
                if not np.isclose(v, 0.0):
                    a = np.zeros(n_dims, dtype=np.float32)
                    a[i] = v
                    actions.append(a)

        self.actions_grid = np.array(actions, dtype=np.float32)
        self.action_space = Discrete(len(self.actions_grid))
        print(f"DiscretizedActionWrapper: {len(self.actions_grid)} acciones discretas.")

    def action(self, action_index):
        idx = int(action_index.item() if hasattr(action_index, "item") else action_index)
        return self.actions_grid[idx]


# ---------------------------------------------------------------------------
# Conversión a escala de grises con OpenCV
# ---------------------------------------------------------------------------
class GrayscaleWrapper(gym.ObservationWrapper):
    """Convierte observaciones RGB a escala de grises."""
    def __init__(self, env):
        super().__init__(env)
        h, w = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=(h, w), dtype=np.uint8)

    def observation(self, obs):
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Wrapper clave: obtiene el render, lo redimensiona y convierte a gris
# ---------------------------------------------------------------------------
class PixelObservationWrapper(gym.ObservationWrapper):
    """
    Reemplaza la observación del entorno por un frame renderizado en gris.
    El entorno debe estar creado con render_mode='rgb_array'.
    El frame se redimensiona a obs_size x obs_size.

    Compatible con:
      - Walker2d-v5 (renderiza directamente a 84x84 si se pasa width/height)
      - CartPole-v1 (renderiza a resolución por defecto, luego resize)
    """
    def __init__(self, env, obs_size=84):
        super().__init__(env)
        self.obs_size = obs_size
        self.observation_space = Box(
            low=0, high=255, shape=(obs_size, obs_size), dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render()          # RGB np.array (H, W, 3)
        if frame is None:
            return np.zeros((self.obs_size, self.obs_size), dtype=np.uint8)
        # Resize si hace falta
        if frame.shape[0] != self.obs_size or frame.shape[1] != self.obs_size:
            frame = cv2.resize(frame, (self.obs_size, self.obs_size),
                               interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Visualización en tiempo real con OpenCV (solo para eval)
# ---------------------------------------------------------------------------
class OpenCVRenderWrapper(gym.Wrapper):
    """Muestra el render en ventana flotante OpenCV (usa solo para evaluación)."""
    def __init__(self, env, window_name="Preview"):
        super().__init__(env)
        self.window_name = window_name

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        img = self.env.render()
        if img is not None:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.resize(img_bgr, (480, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(self.window_name, img_bgr)
            cv2.waitKey(1)
        return obs, reward, terminated, truncated, info

    def close(self):
        cv2.destroyAllWindows()
        return self.env.close()


# ---------------------------------------------------------------------------
# Reward shaping para Walker2d
# ---------------------------------------------------------------------------
class WalkerRewardWrapper(gym.RewardWrapper):
    """
    Penaliza el ángulo del torso, las rodillas hiperestendidas y la asimetría
    de marcha. Mantiene el reward de avance como señal principal.
    """
    def __init__(self, env, torso_w=1.0, knee_w=0.3, sym_w=0.1):
        super().__init__(env)
        self.torso_w = torso_w
        self.knee_w  = knee_w
        self.sym_w   = sym_w
        # Guardamos la última obs para el reward shaping
        self._last_obs = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        if self._last_obs is None:
            return reward
        obs = self._last_obs
        # Índices del espacio de observación de Walker2d-v5:
        # 0: z del torso, 1: ángulo torso, 2: muslo izq, 3: pierna izq,
        # 4: pie izq, 5: muslo der, 6: pierna der, 7: pie der, ...
        torso_angle  = obs[1]
        left_knee    = obs[3]
        right_knee   = obs[6]
        left_thigh   = obs[2]
        right_thigh  = obs[5]

        torso_penalty = self.torso_w * torso_angle ** 2
        knee_penalty  = self.knee_w  * (left_knee ** 2 + right_knee ** 2)
        sym_penalty   = self.sym_w   * (
            (left_knee  - right_knee)  ** 2 +
            (left_thigh - right_thigh) ** 2
        )
        return reward - torso_penalty - knee_penalty - sym_penalty


# ---------------------------------------------------------------------------
# Constructores de entorno
# ---------------------------------------------------------------------------
def make_pixel_env(env_id, render=False, seed=42):
    """
    Pipeline para observaciones en píxeles (gris, 84×84, stack de 4 frames).
    Estructura del stack resultante: (4, 84, 84) — listo para CNN.
    """
    obs_size = 84

    if "Walker2d" in env_id:
        # MuJoCo renderiza directamente al tamaño pedido → sin resize posterior
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            width=obs_size,
            height=obs_size,
            healthy_angle_range=(-0.4, 0.4),
            max_episode_steps=2500,
        )
        env = WalkerRewardWrapper(env)

    elif "CartPole" in env_id:
        # CartPole no acepta width/height → render a resolución nativa, luego resize
        env = gym.make(env_id, render_mode="rgb_array")

    elif "Hopper" in env_id or "HalfCheetah" in env_id:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            width=obs_size,
            height=obs_size,
        )
    else:
        env = gym.make(env_id, render_mode="rgb_array")

    # Wrapper principal: render → gris → (84, 84)
    env = PixelObservationWrapper(env, obs_size=obs_size)

    # Discretización para entornos con acción continua
    if isinstance(env.action_space, Box):
        env = DiscretizedActionWrapper(env, bins=3)

    # Stack de 4 frames: (84, 84) × 4 → (4, 84, 84)
    env = FrameStackObservation(env, stack_size=4)

    if render:
        env = OpenCVRenderWrapper(env)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_state_env(env_id, render=False, seed=42):
    """Entorno con observación de estado vectorial."""
    if "Walker2d" in env_id:
        env = gym.make(
            env_id,
            render_mode="human" if render else None,
            healthy_angle_range=(-0.4, 0.4),
            max_episode_steps=2500,
        )
        env = WalkerRewardWrapper(env)
    else:
        env = gym.make(env_id, render_mode="human" if render else None)

    if isinstance(env.action_space, Box):
        env = DiscretizedActionWrapper(env, bins=3)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_env(env_id, obs_type, render=False, seed=42):
    if obs_type == "pixel":
        return make_pixel_env(env_id, render, seed)
    elif obs_type == "state":
        return make_state_env(env_id, render, seed)
    else:
        raise ValueError(f"obs_type desconocido: {obs_type}")


def make_vec_env(env_id, obs_type, num_envs, seed=42):
    """
    Crea un entorno vectorizado con num_envs entornos en paralelo.
    Usa SyncVectorEnv (más seguro con MuJoCo en múltiples procesos).
    """
    def _make_thunk(idx):
        def _thunk():
            return make_env(env_id, obs_type, render=False, seed=seed + idx)
        return _thunk

    thunks = [_make_thunk(i) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(thunks)