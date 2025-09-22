import os
from datetime import datetime
import pandas as pd
import time

# ✅ Use consistent data directory
DATA_DIR = "data"
MODEL_DIR = os.path.join("backend", "model")

def detect_dashboard():
    """Auto-detect which dashboard data is available: moon or mars."""
    for dashboard in ["moon", "mars"]:
        for ext in [".csv", ".xlsx"]:
            path = os.path.join(DATA_DIR, f"{dashboard}_data{ext}")
            if os.path.exists(path):
                print(f"[AUTO DETECT] Found data for: {dashboard}")
                return dashboard
    print("[AUTO DETECT] No dashboard data found.")
    return None

def save_file(file, dashboard):
    """Save uploaded CSV or Excel file to the appropriate dashboard path."""
    os.makedirs(DATA_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".csv", ".xlsx"]:
        raise ValueError("Unsupported file type. Only .csv and .xlsx are allowed.")

    filename = f"{dashboard}_data{ext}"
    path = os.path.join(DATA_DIR, filename)
    file.save(path)
    print(f"[UPLOAD] Saved file to: {path}")
    return path

def get_last_update(dashboard=None):
    """Return the last modified time of the trained model zip file for a given dashboard."""
    if dashboard is None:
        dashboard = detect_dashboard() or "moon"

    model_path = os.path.join(MODEL_DIR, f"{dashboard}_model.zip")
    if os.path.exists(model_path):
        timestamp = os.path.getmtime(model_path)
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    print(f"[LAST UPDATE] No model file found for {dashboard}")
    return "Unavailable"

def load_dashboard_data(dashboard=None, numeric_only=True):
    """Load the latest CSV or Excel file for the given dashboard."""
    if dashboard is None:
        dashboard = detect_dashboard() or "moon"

    for ext in [".csv", ".xlsx"]:
        path = os.path.join(DATA_DIR, f"{dashboard}_data{ext}")
        if os.path.exists(path):
            try:
                if ext == ".csv":
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)

                df.columns = [col.lower().strip() for col in df.columns]  # Normalize column names

                if numeric_only:
                    df = df.select_dtypes(include=["number"])
                    if df.empty:
                        print(f"[DATA WARNING] No numeric features found in {path}")
                    else:
                        print(f"[DATA LOADED] {dashboard} numeric features: {df.columns.tolist()}")
                else:
                    print(f"[DATA LOADED] {dashboard} full dataset with columns: {df.columns.tolist()}")

                return df
            except Exception as e:
                print(f"[DATA ERROR] Failed to load {path}: {e}")
                return pd.DataFrame()

    print(f"[DATA MISSING] No data file found for {dashboard} dashboard.")
    return pd.DataFrame()
