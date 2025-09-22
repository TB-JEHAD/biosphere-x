from stable_baselines3 import PPO
from biosphere_env import BiosphereEnv

import os
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # ✅ Use non-GUI backend
import matplotlib.pyplot as plt

DATA_DIR = "data"
MODEL_DIR = os.path.join("backend", "model")

def train_model(dashboard):
    try:
        # ✅ Initialize environment
        env = BiosphereEnv(dashboard)
        if not isinstance(env, gym.Env):
            raise TypeError("BiosphereEnv must inherit from gymnasium.Env")

        # ✅ Train PPO model
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)

        # ✅ Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{dashboard}_model.zip")
        model.save(model_path)

        # ✅ Save dummy learning rate plot
        plot_path = os.path.join(MODEL_DIR, f"{dashboard}_learning_rate.png")
        plt.figure(figsize=(6, 3))
        plt.plot([0, 5000, 10000], [0.0003, 0.00025, 0.0002], label="Learning Rate")
        plt.xlabel("Timesteps")
        plt.ylabel("Rate")
        plt.title(f"{dashboard.capitalize()} AI Learning Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"[TRAIN] Model saved to: {model_path}")
        print(f"[TRAIN] Learning rate plot saved to: {plot_path}")

        return {
            "status": "success",
            "model_path": model_path,
            "plot_path": plot_path,
            "message": f"{dashboard.capitalize()} model training complete."
        }

    except Exception as e:
        print(f"[TRAIN ERROR] Dashboard: {dashboard}, Reason: {e}")
        return {"error": str(e)}

# ✅ Auto-detect dashboards and train
if __name__ == "__main__":
    trained = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv") or filename.endswith(".xlsx"):
            dashboard = filename.split("_data")[0].lower()
            result = train_model(dashboard)
            trained.append({dashboard: result})

    print("[AUTO TRAIN COMPLETE]")
    for item in trained:
        print(item)

