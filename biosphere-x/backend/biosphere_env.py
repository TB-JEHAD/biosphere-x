import gymnasium as gym
import numpy as np
import pandas as pd
from utils import load_dashboard_data

class BiosphereEnv(gym.Env):
    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.data = load_dashboard_data(dashboard)

        # ✅ Filter numeric columns only
        self.data = self.data.select_dtypes(include=["number"])

        # ✅ Drop rows with missing values
        before = len(self.data)
        self.data = self.data.dropna()
        after = len(self.data)
        print(f"[ENV CLEAN] Dropped {before - after} rows with missing values for {dashboard}")

        # ✅ Separate features and labels
        self.features = self.data.drop(columns=["target"], errors="ignore").values
        self.labels = self.data["target"].values if "target" in self.data.columns else np.zeros(len(self.data))

        # ✅ Validate feature shape
        if self.features.size == 0:
            raise ValueError(f"[ENV ERROR] No valid numeric features found for dashboard: {dashboard}")

        # ✅ Normalize features to [0, 1]
        self.features = self._normalize(self.features)

        # ✅ Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)  # 0 = not viable, 1 = viable

        self.index = 0

    def _normalize(self, data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        ranges = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        return (data - min_vals) / ranges

    def reset(self, seed=None, options=None):
        self.index = 0
        return self.features[self.index], {}

    def step(self, action):
        label = self.labels[self.index]
        reward = 1 if action == label else -1
        self.index += 1
        done = self.index >= len(self.features)
        obs = self.features[self.index] if not done else np.zeros_like(self.features[0])
        return obs, reward, done, False, {}
