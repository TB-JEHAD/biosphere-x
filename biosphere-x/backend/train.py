from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from biosphere_env import BiosphereEnv

import os
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # ✅ Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data"
MODEL_DIR = os.path.join("backend", "model")

class LearningRateCallback(BaseCallback):
    """
    Callback to track learning rate during training
    """
    def __init__(self, verbose=0):
        super(LearningRateCallback, self).__init__(verbose)
        self.learning_rates = []
        self.timesteps = []
    
    def _on_step(self):
        # Get current learning rate from the optimizer
        current_lr = self.model.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        self.timesteps.append(self.num_timesteps)
        return True

def train_model(dashboard):
    try:
        # ✅ Initialize environment
        env = BiosphereEnv(dashboard)
        if not isinstance(env, gym.Env):
            raise TypeError("BiosphereEnv must inherit from gymnasium.Env")

        # ✅ Create callback to track learning rate
        lr_callback = LearningRateCallback()
        
        # ✅ Train PPO model with learning rate tracking
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=50000, callback=lr_callback)

        # ✅ Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{dashboard}_model.zip")
        model.save(model_path)

        # ✅ Create real learning rate plot
        plot_path = os.path.join(MODEL_DIR, f"{dashboard}_learning_rate.png")
        
        plt.figure(figsize=(8, 4))
        
        if len(lr_callback.learning_rates) > 0:
            # Plot actual learning rate data
            plt.plot(lr_callback.timesteps, lr_callback.learning_rates, 
                    label="Actual Learning Rate", linewidth=2, color='blue')
            
            # Add some statistics
            mean_lr = np.mean(lr_callback.learning_rates)
            min_lr = np.min(lr_callback.learning_rates)
            max_lr = np.max(lr_callback.learning_rates)
            
            plt.axhline(y=mean_lr, color='red', linestyle='--', 
                       alpha=0.7, label=f'Mean LR: {mean_lr:.6f}')
            
            plt.fill_between(lr_callback.timesteps, min_lr, max_lr, 
                           alpha=0.2, color='blue', label=f'Range: {min_lr:.6f} - {max_lr:.6f}')
            
            plt.xlabel("Timesteps")
            plt.ylabel("Learning Rate")
            plt.title(f"{dashboard.capitalize()} AI - Actual Learning Rate Progression")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add text box with statistics
            stats_text = f"""Training Statistics:
Total Steps: {lr_callback.timesteps[-1] if lr_callback.timesteps else 0}
Final LR: {lr_callback.learning_rates[-1]:.8f if lr_callback.learning_rates else 0}
LR Changes: {len(set(lr_callback.learning_rates))}"""
            
            plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=8, verticalalignment='bottom')
        else:
            # Fallback if no learning rate data was collected
            plt.text(0.5, 0.5, "No learning rate data collected", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.xlabel("Timesteps")
            plt.ylabel("Learning Rate")
            plt.title(f"{dashboard.capitalize()} AI - Learning Rate Tracking")
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[TRAIN] Model saved to: {model_path}")
        print(f"[TRAIN] Real learning rate plot saved to: {plot_path}")
        print(f"[TRAIN] Learning rate statistics:")
        if lr_callback.learning_rates:
            print(f"  - Initial LR: {lr_callback.learning_rates[0]:.6f}")
            print(f"  - Final LR: {lr_callback.learning_rates[-1]:.6f}")
            print(f"  - Mean LR: {np.mean(lr_callback.learning_rates):.6f}")
            print(f"  - Unique LR values: {len(set(lr_callback.learning_rates))}")

        return {
            "status": "success",
            "model_path": model_path,
            "plot_path": plot_path,
            "learning_rate_stats": {
                "initial": float(lr_callback.learning_rates[0]) if lr_callback.learning_rates else 0,
                "final": float(lr_callback.learning_rates[-1]) if lr_callback.learning_rates else 0,
                "mean": float(np.mean(lr_callback.learning_rates)) if lr_callback.learning_rates else 0,
                "min": float(np.min(lr_callback.learning_rates)) if lr_callback.learning_rates else 0,
                "max": float(np.max(lr_callback.learning_rates)) if lr_callback.learning_rates else 0
            },
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