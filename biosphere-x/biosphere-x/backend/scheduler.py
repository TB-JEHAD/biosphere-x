from apscheduler.schedulers.background import BackgroundScheduler
from train import train_model
from train_microbe_model import train_microbe_model

scheduler = BackgroundScheduler()

# ✅ Schedule PPO model training
scheduler.add_job(lambda: train_model("mars"), 'interval', hours=6)
scheduler.add_job(lambda: train_model("moon"), 'interval', hours=6)

# ✅ Schedule microbe model training
scheduler.add_job(lambda: train_microbe_model("mars"), 'interval', hours=6)
scheduler.add_job(lambda: train_microbe_model("moon"), 'interval', hours=6)

scheduler.start()
