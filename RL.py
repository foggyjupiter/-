import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class RainPiezoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, length, theta_max=np.deg2rad(60.0)):
        super().__init__()
        self.length = length                # 점 개수
        self.n_seg = length - 1             # 세그먼트 개수
        self.dx = 1.0 / self.n_seg          # 격자 간격
        self.theta_max = float(theta_max)    # 각도 제한
        self.maxsteps = 10000
        self.currnetstep = 0

        # 상태: 각 세그먼트의 각도 배열
        self.observation_space = spaces.Box(
            low=np.full((self.n_seg,), 0.0, dtype=np.float32),
            high=np.full((self.n_seg,), self.theta_max, dtype=np.float32),
            shape=(self.n_seg,), dtype = np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([0, -np.deg2rad(2.0)], dtype=np.float32),
            high=np.array([self.n_seg - 1, np.deg2rad(2.0)], dtype=np.float32),
            dtype=np.float32
        )

        self.angles = None    # 관찰(각도) 상태

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #초기 조건
        self.angles = np.zeros(self.n_seg, dtype=np.float32)
        self.currnetstep = 0
        return self.angles.copy(), {}

    def step(self, action):
        self.currnetstep += 1
        idx_f, delta_theta = action
        idx = int(np.clip(idx_f, 0, self.n_seg - 1))
        #세타를 바꿈
        new_theta = np.clip(self.angles[idx] + float(delta_theta), 0.0, self.theta_max)
        self.angles[idx] = new_theta

        reward = self._compute_reward()
        terminated = False
        truncated = self.currnetstep >= self.maxsteps
        return self.angles.copy(), float(reward), terminated, truncated, {}

    def _compute_reward(self):
        #보상함수
        return 0.0

    def render(self):
        print("angles(rad):", self.angles.tolist())

    def close(self):
        pass

if __name__ == "__main__":
    env = RainPiezoEnv(length=10)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)

    obs, info = env.reset()
    for _ in range(20):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
