import pandas as pd
import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

def train_microbe_model(dashboard):
    path = f"data/{dashboard}_data.xlsx"
    df = pd.read_excel(path)

    # ✅ Choose features based on dashboard
    feature_cols = [
        "Min Temperature (°C)", "Max Temperature (°C)", "Humidity (%)",
        "Radiation (mSv/day)", "Perchlorates", "Sulfates",
        "Ice Depth (m)", "Habitability Score"
    ] if dashboard == "mars" else [
        "Min Temperature (°C)", "Max Temperature (°C)", "Radiation (mSv/day)",
        "Water Ice Present", "Ilmenite Content", "Regolith Depth (m)",
        "Sunlight Hours", "Habitability Score"
    ]

    label_col = "Suitable Microorganisms"
    df = df.dropna(subset=feature_cols + [label_col])

    # ✅ Encode microbe labels
    df[label_col] = df[label_col].apply(lambda x: [m.strip() for m in str(x).split(",") if m.strip()])
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df[label_col])

    # ✅ Encode features
    X = pd.get_dummies(df[feature_cols])
    X = X.fillna(X.mean())

    # ✅ Train model
    model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
    model.fit(X, Y)

    # ✅ Save model and label encoder
    model_dir = "backend/model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/{dashboard}_microbe_model.pkl")
    joblib.dump(mlb, f"{model_dir}/{dashboard}_mlb.pkl")

    print(f"[MICROBE TRAIN] {dashboard} model trained with {len(mlb.classes_)} microbes.")
if __name__ == "__main__":
    for dashboard in ["mars", "moon"]:
        train_microbe_model(dashboard)
