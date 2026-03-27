import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import FrameStackObservation


# ---------------------------------------------------------------------------
# Discretización de acciones continuas
# ---------------------------------------------------------------------------
class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=3):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), \
            "DiscretizedActionWrapper requiere espacio de acciones Box."

        low = self.env.action_space.low
        high = self.env.action_space.high
        n_dims = self.env.action_space.shape[0]

        actions = [np.zeros(n_dims, dtype=np.float32)]
        for i in range(n_dims):
            values = np.linspace(low[i], high[i], bins)
            for v in values:
                if not np.isclose(v, 0.0):
                    a = np.zeros(n_dims, dtype=np.float32)
                    a[i] = v
                    actions.append(a)

        self.actions_grid = np.array(actions, dtype=np.float32)
        self.action_space = Discrete(len(self.actions_grid))
        print(f"DiscretizedActionWrapper: {len(self.actions_grid)} discrete actions.")

    def action(self, action_index):
        idx = int(action_index.item() if hasattr(action_index, "item") else action_index)
        return self.actions_grid[idx]


# ---------------------------------------------------------------------------
# Pixel observation: render → grayscale → resize
# ---------------------------------------------------------------------------
class PixelObservationWrapper(gym.ObservationWrapper):
    """Replace obs with grayscale render. Works for envs that don't support
    width/height in gym.make (e.g. CartPole)."""

    def __init__(self, env, obs_size=84):
        super().__init__(env)
        self.obs_size = obs_size
        self.observation_space = Box(
            low=0, high=255, shape=(obs_size, obs_size), dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render()
        if frame is None:
            return np.zeros((self.obs_size, self.obs_size), dtype=np.uint8)
        if frame.shape[0] != self.obs_size or frame.shape[1] != self.obs_size:
            frame = cv2.resize(frame, (self.obs_size, self.obs_size),
                               interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# RenderGrayscaleWrapper (for MuJoCo envs that support width/height)
# ---------------------------------------------------------------------------
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
        frame = self.env.render()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if gray.shape != (self._obs_size, self._obs_size):
            gray = cv2.resize(
                gray, (self._obs_size, self._obs_size), interpolation=cv2.INTER_AREA
            )
        return gray


# ---------------------------------------------------------------------------
# EvalRenderWrapper with video recording
# ---------------------------------------------------------------------------
class EvalRenderWrapper(gym.Wrapper):
    """High-quality display for eval with video recording support."""

    def __init__(self, env, display_size=480, window_name="Eval"):
        super().__init__(env)
        self._display_size = display_size
        self._window_name = window_name
        self._mj_renderer = None
        self._camera = None
        self._episode_frames = []
        self._is_mujoco = hasattr(env.unwrapped, "model") and hasattr(
            env.unwrapped, "data"
        )

    def _init_mj_renderer(self):
        if self._mj_renderer is not None:
            return
        import mujoco

        unwrapped = self.env.unwrapped
        unwrapped.model.vis.global_.offwidth = max(
            unwrapped.model.vis.global_.offwidth, self._display_size
        )
        unwrapped.model.vis.global_.offheight = max(
            unwrapped.model.vis.global_.offheight, self._display_size
        )
        self._mj_renderer = mujoco.Renderer(
            unwrapped.model,
            height=self._display_size,
            width=self._display_size,
        )
        mj_rend = getattr(unwrapped, "mujoco_renderer", None)
        if mj_rend is not None and getattr(mj_rend, "viewer", None) is None:
            mj_rend.render("rgb_array")
        viewer = getattr(mj_rend, "viewer", None) if mj_rend else None
        if viewer is not None and hasattr(viewer, "cam"):
            src = viewer.cam
            self._camera = mujoco.MjvCamera()
            self._camera.type = src.type
            self._camera.fixedcamid = src.fixedcamid
            self._camera.trackbodyid = src.trackbodyid
            self._camera.distance = src.distance
            self._camera.azimuth = src.azimuth
            self._camera.elevation = src.elevation
            self._camera.lookat[:] = src.lookat
        else:
            self._camera = -1

    def _get_display_frame(self):
        if self._is_mujoco:
            self._init_mj_renderer()
            self._mj_renderer.update_scene(
                self.env.unwrapped.data, camera=self._camera
            )
            return self._mj_renderer.render()
        frame = self.env.render()
        if frame is not None:
            h, w = frame.shape[:2]
            if h != self._display_size or w != self._display_size:
                frame = cv2.resize(
                    frame,
                    (self._display_size, self._display_size),
                    interpolation=cv2.INTER_LINEAR,
                )
        return frame

    def _show(self, frame):
        if frame is None:
            return
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(self._window_name, img_bgr)
        cv2.waitKey(1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._get_display_frame()
        if frame is not None:
            self._episode_frames.append(frame)
        self._show(frame)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._episode_frames = []
        result = self.env.reset(**kwargs)
        frame = self._get_display_frame()
        if frame is not None:
            self._episode_frames.append(frame)
        self._show(frame)
        return result

    def get_episode_frames(self):
        return self._episode_frames

    def save_video(self, path, fps=None):
        """Save recorded episode frames as H.264 MP4 via imageio-ffmpeg."""
        if not self._episode_frames:
            return
        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        import imageio.v3 as iio
        iio.imwrite(path, self._episode_frames, fps=fps, codec="libx264")

    def close(self):
        if self._mj_renderer is not None:
            self._mj_renderer.close()
            self._mj_renderer = None
        cv2.destroyAllWindows()
        return self.env.close()


# ---------------------------------------------------------------------------
# Reward shaping for Walker2d
# ---------------------------------------------------------------------------
class WalkerRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, torso_w=1.0, knee_w=0.3, sym_w=0.1):
        super().__init__(env)
        self.torso_w = torso_w
        self.knee_w = knee_w
        self.sym_w = sym_w
        self._last_obs = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        if self._last_obs is None:
            return reward
        obs = self._last_obs
        torso_angle = obs[1]
        left_knee = obs[3]
        right_knee = obs[6]
        left_thigh = obs[2]
        right_thigh = obs[5]
        torso_penalty = self.torso_w * torso_angle ** 2
        knee_penalty = self.knee_w * (left_knee ** 2 + right_knee ** 2)
        sym_penalty = self.sym_w * (
            (left_knee - right_knee) ** 2 +
            (left_thigh - right_thigh) ** 2
        )
        return reward - torso_penalty - knee_penalty - sym_penalty


# ---------------------------------------------------------------------------
# Environment constructors
# ---------------------------------------------------------------------------
def make_pixel_env(env_id, render=False, seed=42):
    obs_size = 84

    if "Walker2d" in env_id:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            width=obs_size,
            height=obs_size,
            healthy_angle_range=(-0.4, 0.4),
            max_episode_steps=2500,
        )
        env = WalkerRewardWrapper(env)
        env = RenderGrayscaleWrapper(env, obs_size=obs_size)
    elif "CartPole" in env_id:
        env = gym.make(env_id, render_mode="rgb_array")
        env = PixelObservationWrapper(env, obs_size=obs_size)
    elif "Hopper" in env_id or "HalfCheetah" in env_id:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            width=obs_size,
            height=obs_size,
        )
        env = RenderGrayscaleWrapper(env, obs_size=obs_size)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
        env = PixelObservationWrapper(env, obs_size=obs_size)

    if isinstance(env.action_space, Box):
        env = DiscretizedActionWrapper(env, bins=3)

    env = FrameStackObservation(env, stack_size=4)

    if render:
        env = EvalRenderWrapper(env)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_state_env(env_id, render=False, seed=42):
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
        raise ValueError(f"Unsupported obs_type: {obs_type}")


def make_vec_env(env_id, obs_type, num_envs, seed=42):
    """Create a vectorized environment with num_envs parallel envs."""
    def _make_thunk(idx):
        def _thunk():
            return make_env(env_id, obs_type, render=False, seed=seed + idx)
        return _thunk

    thunks = [_make_thunk(i) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(thunks)
