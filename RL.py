import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RainPiezoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, length):
        super(RainPiezoEnv, self).__init__()

        self.length = length  # x축 쪼개기

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.length,), dtype=np.float32) # observation: x좌표별 높이를 제한과 함께 줌
        
        self.action_space = spaces.Box( # action: 특정 인덱스를 고르고 높이를 조정
            low=np.array([0, -0.05]),
            high=np.array([self.length-1, 0.05]),
            dtype=np.float32
        )
        
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.linspace(0, 1, self.length).astype(np.float32) #일차함수(초기값)
        return self.state, {}

    def step(self, action):
        idx, delta = action
        idx = int(np.clip(idx, 0, self.length-1))
        
        self.state[idx] = np.clip(self.state[idx] + delta, 0.0, 1.0)
        self.state = np.maximum.accumulate(self.state)

        reward = self._compute_reward()

        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def _compute_reward(self): #보상함수
        reward = 0
        return float(reward)

    def render(self):
        # import matplotlib.pyplot as plt
        # plt.plot(self.state)
        # plt.show()
        print(list(self.state))

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
