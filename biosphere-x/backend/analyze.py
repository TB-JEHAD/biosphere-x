from stable_baselines3 import PPO
from biosphere_env import BiosphereEnv
import numpy as np
import os

DATA_DIR = "data"
MODEL_DIR = os.path.join("backend", "model")

def analyze_data(dashboard):
    try:
        model_path = os.path.join(MODEL_DIR, f"{dashboard}_model.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        model = PPO.load(model_path)
        env = BiosphereEnv(dashboard)
        obs, _ = env.reset()

        predictions = []
        confidences = []

        if not hasattr(env, "features") or env.features is None or not env.features.any():
            raise ValueError("Environment has no valid features to analyze.")

        for _ in range(len(env.features)):
            action, prob = model.predict(obs, deterministic=False)

            # ✅ Safe confidence extraction
            try:
                if isinstance(prob, (list, np.ndarray)):
                    confidence = float(np.max(prob))
                elif isinstance(prob, (float, int)):
                    confidence = float(prob)
                else:
                    confidence = 0.0
            except Exception:
                confidence = 0.0

            predictions.append(int(action))
            confidences.append(confidence)

            obs, _, done, _, _ = env.step(action)
            if done:
                break

        viable_count = sum(predictions)
        total = len(predictions)
        viability_ratio = viable_count / total if total > 0 else 0.0
        last_confidence = round(confidences[-1], 3) if confidences else 0.0

        return {
            "dashboard": dashboard,
            "total_samples": total,
            "viable_predictions": viable_count,
            "viability_ratio": round(viability_ratio, 3),
            "last_prediction_confidence": last_confidence,
            "message": f"{dashboard.capitalize()} viability analysis complete."
        }

    except Exception as e:
        print(f"[ANALYZE ERROR] Dashboard: {dashboard}, Reason: {e}")
        return {"dashboard": dashboard, "error": str(e)}

# ✅ Auto-detect dashboards and analyze
if __name__ == "__main__":
    analyzed = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv") or filename.endswith(".xlsx"):
            dashboard = filename.split("_data")[0].lower()
            result = analyze_data(dashboard)
            analyzed.append(result)

    print("[AUTO ANALYSIS COMPLETE]")
    for item in analyzed:
        print(item)
