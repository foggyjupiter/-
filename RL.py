# shape_opt_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def reward_from_profile(f: np.ndarray) -> float:
    #보상함수
    #입력: f (길이 N의 높이 배열, 단조증가가 보장됨)
    #반환: float (스칼라 보상)
    #추후 짜야하는 부분 (물리학적 해석)


class ShapeOptimizeEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_points: int = 100,
        f_min: float = 0.0,
        f_max: float = 1.0,
        delta: float = 0.02,
        episode_len: int = 64,
        monotone: bool = True,
        action_mode: str = "discrete_index",  # 'discrete_index' | 'continuous_vector'
        reward_fn = reward_from_profile,
        seed: int | None = None,
    ):
        super().__init__()
        self.n = int(n_points)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.delta = float(delta)
        self.episode_len = int(episode_len)
        self.monotone = bool(monotone)
        self.action_mode = action_mode
        if action_mode not in ("discrete_index", "continuous_vector"):
            raise ValueError("action_mode must be 'discrete_index' or 'continuous_vector'")

        if reward_fn is None:
            raise ValueError("reward_fn must be provided")
        self.reward_fn = reward_fn

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self.f = None

        # Observation space
        self.observation_space = spaces.Box(
            low=self.f_min, high=self.f_max, shape=(self.n,), dtype=np.float32
        )

        # Action space
        if self.action_mode == "discrete_index":
            # 2N: (i, +Δ) 또는 (i, -Δ)
            self.action_space = spaces.Discrete(2 * self.n)
        else:
            # [-1, 1]^N  -> f += delta * a
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

    # ----- 유틸 -----
    def _project_monotone(self, x: np.ndarray) -> np.ndarray:
        # 비감소(non-decreasing) 강제
        return np.maximum.accumulate(x)

    def _clip_profile(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.f_min, self.f_max)

    # ----- Gym API -----
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        # 초기 프로파일: 선형 + 약간의 잡음
        base = np.linspace(self.f_min, self.f_max, self.n)
        noise = self._rng.normal(0.0, 0.01, size=self.n)
        f0 = base + noise
        if self.monotone:
            f0 = self._project_monotone(f0)
        self.f = self._clip_profile(f0)
        return self.f.astype(np.float32), {}

    def step(self, action):
        self._t += 1

        if self.action_mode == "discrete_index":
            if not isinstance(action, (int, np.integer)):
                # SB3가 np.array(int)로 줄 때가 있어 형 변환
                action = int(action)
            idx = action // 2
            sign = +1.0 if (action % 2) == 0 else -1.0
            idx = int(np.clip(idx, 0, self.n - 1))
            self.f[idx] += sign * self.delta
        else:
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            self.f = self.f + self.delta * action

        # 경계/단조 보정
        self.f = self._clip_profile(self.f)
        if self.monotone:
            self.f = self._project_monotone(self.f)
            self.f = self._clip_profile(self.f)  # 상한 넘어갔을 경우 재클립

        # 보상 계산 (네 보상함수 호출)
        reward = float(self.reward_fn(self.f))

        terminated = False
        truncated = self._t >= self.episode_len
        info = {}
        return self.f.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        # 필요시 디버그용
        print(f"[t={self._t}] f[:5]={self.f[:5]} ...")


# =========================
# (2) 학습 스켈레톤 (PPO)
# =========================
if __name__ == "__main__":
    # 여기에 네 보상함수 구현해도 되고, 외부에서 import해도 됨.
    # 예시:
    # def reward_from_profile(f):
    #     # 예: 기울기 절댓값 평균 (임시)
    #     s = np.gradient(f, edge_order=2)
    #     return float(np.mean(np.abs(s)))

    from stable_baselines3 import PPO

    # --- 환경 생성 ---
    env = ShapeOptimizeEnv(
        n_points=100,
        f_min=0.0,
        f_max=1.0,
        delta=0.02,
        episode_len=64,
        monotone=True,
        action_mode="discrete_index",  # 'continuous_vector'로 바꿔도 됨
        reward_fn=reward_from_profile, # 반드시 구현해야 함
        seed=42,
    )

    # --- 알고리즘 선택: PPO (discrete/continuous 모두 지원) ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=2048,
        n_steps=2048,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        n_epochs=10,
        device="auto",
        tensorboard_log=None,
    )

    # --- 학습 ---
    # 보상함수 미구현이면 여기서 NotImplementedError 발생함 (의도된 동작)
    model.learn(total_timesteps=50_000)

    # --- 평가/롤아웃 ---
    obs, _ = env.reset(seed=123)
    total_reward = 0.0
    for t in range(64):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        total_reward += r
        if term or trunc:
            break

    print("Rollout total reward:", total_reward)
    print("Final profile (first 10):", obs[:10])