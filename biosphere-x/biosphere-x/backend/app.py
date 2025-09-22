from flask import Flask, request, jsonify
from train import train_model
from analyze import analyze_data
from utils import save_file, get_last_update, load_dashboard_data
import subprocess
import threading
import time
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
import os
import re
from train_microbe_model import train_microbe_model

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# 🔧 Launch frontend server
def start_frontend():
    time.sleep(1)
    subprocess.Popen(["python", "-m", "http.server", "8000"], cwd="../frontend")

# 📦 Upload data file
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        dashboard = request.form['dashboard']
        path = save_file(file, dashboard)
        print(f"[UPLOAD] Saved file to: {path}")
        return jsonify({"status": "uploaded", "path": path})
    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# 🔍 Analyze biosphere data
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        dashboard = request.form['dashboard']
        model_path = os.path.join("backend", "model", f"{dashboard}_model.zip")
        microbe_model_path = os.path.join("backend", "model", f"{dashboard}_microbe_model.pkl")

        # ✅ Train PPO model if missing
        if not os.path.exists(model_path):
            print(f"[ANALYZE] No PPO model found for {dashboard}. Training now...")
            train_model(dashboard)

        # ✅ Train microbe model if missing
        if not os.path.exists(microbe_model_path):
            print(f"[ANALYZE] No microbe model found for {dashboard}. Training now...")
            train_microbe_model(dashboard)

        # ✅ Run analysis
        result = analyze_data(dashboard)
        print(f"[ANALYZE] Dashboard: {dashboard}, Result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"[ANALYZE ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# 🔁 Retrain AI model
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        dashboard = request.form['dashboard']
        print(f"[RETRAIN] Starting retraining for {dashboard}...")
        
        # Train both models
        train_model(dashboard)
        train_microbe_model(dashboard)
        
        # Train enhanced chatbot
        try:
            from chatbot_trainer import BiosphereChatbotTrainer
            trainer = BiosphereChatbotTrainer()
            trainer.generate_knowledge_base(dashboard)
            trainer.save_trained_model(dashboard)
            print(f"[RETRAIN] Enhanced chatbot trained for {dashboard}")
        except Exception as e:
            print(f"[RETRAIN] Chatbot training failed for {dashboard}: {e}")
        
        print(f"[RETRAIN] Models retrained for dashboard: {dashboard}")
        return jsonify({"status": "retrained", "message": f"{dashboard.capitalize()} models retrained successfully!"})
    except Exception as e:
        print(f"[RETRAIN ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# 🤖 Train Enhanced Chatbot
@app.route('/train-chatbot', methods=['POST'])
def train_chatbot():
    try:
        dashboard = request.form.get('dashboard', 'all')
        print(f"[CHATBOT TRAIN] Starting chatbot training for {dashboard}...")
        
        from chatbot_trainer import BiosphereChatbotTrainer
        trainer = BiosphereChatbotTrainer()
        
        if dashboard == 'all':
            trainer.train_all_dashboards()
            return jsonify({"message": "All chatbot models trained successfully!"})
        else:
            trainer.generate_knowledge_base(dashboard)
            trainer.save_trained_model(dashboard)
            return jsonify({"message": f"{dashboard.capitalize()} chatbot trained successfully!"})
            
    except Exception as e:
        print(f"[CHATBOT TRAIN ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# 📊 Serve chart data
@app.route('/get-chart-data')
def get_chart_data():
    dashboard = request.args.get('dashboard')
    metric = request.args.get('metric')

    try:
        df = load_dashboard_data(dashboard, numeric_only=False)
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"[CHART] No data available for dashboard '{dashboard}'")
            return jsonify({"timestamps": [], "actual": [], "predicted": []})

        def normalize(col):
            return re.sub(r"[^\w]", "", col.lower())

        # Map frontend metric names to actual column names
        metric_mapping = {
            "temperature": ["mintemperaturec"],
            "pressure": ["pressurepa"],
            "wind": ["windspeedms"],
            "co2": ["atmosphericco2"],
            "humidity": ["humidity"],
            "sunlight": ["sunlighthours"]
        }

        normalized_metric = normalize(metric)
        matched_col = None

        # First try direct mapping
        if normalized_metric in metric_mapping:
            for target in metric_mapping[normalized_metric]:
                for col in df.columns:
                    if normalize(col) == target:
                        matched_col = col
                        print(f"[CHART DEBUG] Direct match found: '{col}' -> '{target}'")
                        break
                if matched_col:
                    break
            

        # If no direct mapping, try fuzzy matching
        if not matched_col:
            for col in df.columns:
                if normalized_metric in normalize(col) or normalize(col) in normalized_metric:
                    matched_col = col
                    print(f"[CHART DEBUG] Fuzzy match found: '{col}' -> '{normalize(col)}'")
                    break

        # Debug: Print available columns if no match
        if not matched_col:
            print(f"[CHART DEBUG] No match for '{normalized_metric}'. Available columns: {[normalize(col) for col in df.columns]}")

        if not matched_col:
            print(f"[CHART] Metric '{metric}' not found in dashboard '{dashboard}'. Available columns: {df.columns.tolist()}")
            return jsonify({"timestamps": [], "actual": [], "predicted": []})

        # Use Earth Date as timestamp, fallback to index
        if "Earth Date" in df.columns:
            timestamps = df["Earth Date"].astype(str).tolist()
        else:
            timestamps = list(range(len(df)))

        actual = df[matched_col].tolist()
        # Remove NaN values and create corresponding timestamps
        valid_indices = [i for i, val in enumerate(actual) if pd.notna(val)]
        actual = [actual[i] for i in valid_indices]
        timestamps = [timestamps[i] for i in valid_indices]
        
        # Generate dummy predictions (slightly offset from actual values)
        predicted = [val + (val * 0.1) for val in actual]

        print(f"[CHART] Served metric '{matched_col}' for dashboard '{dashboard}' - {len(actual)} data points")
        return jsonify({"timestamps": timestamps, "actual": actual, "predicted": predicted})
    except Exception as e:
        print(f"[CHART ERROR] {e}")
        return jsonify({"error": str(e), "timestamps": [], "actual": [], "predicted": []}), 500

# 🧠 Enhanced Chat response with AI training
@app.route('/chat-response', methods=['POST'])
def chat_response():
    try:
        data = request.json
        dashboard = data.get("dashboard", "mars")
        question = data.get("question", "").strip()

        print(f"[CHAT] Dashboard: {dashboard}, Question: {question}")

        # Import the enhanced chatbot responder
        try:
            from chatbot_responder import get_chatbot_response
            
            # Get live data insights for contextual responses
            insights = None
            try:
                df = load_dashboard_data(dashboard, numeric_only=False)
                if not df.empty:
                    # Extract basic insights for live data enhancement
                    insights = {
                        'environmental_patterns': {},
                        'habitability_analysis': {},
                        'microorganism_data': {}
                    }
                    
                    # Calculate basic statistics for key metrics
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if col not in ['site id', 'latitude', 'longitude']:
                            insights['environmental_patterns'][col] = {
                                'mean': float(df[col].mean()),
                                'std': float(df[col].std()),
                                'min': float(df[col].min()),
                                'max': float(df[col].max()),
                                'trend': 'stable'  # Simplified for now
                            }
                    
                    # Habitability analysis
                    if 'habitability score' in df.columns:
                        habitability_data = df['habitability score'].dropna()
                        insights['habitability_analysis'] = {
                            'average_score': float(habitability_data.mean()),
                            'score_range': {
                                'min': float(habitability_data.min()),
                                'max': float(habitability_data.max())
                            },
                            'score_distribution': {
                                'excellent': len(habitability_data[habitability_data >= 0.8]),
                                'good': len(habitability_data[(habitability_data >= 0.6) & (habitability_data < 0.8)]),
                                'moderate': len(habitability_data[(habitability_data >= 0.4) & (habitability_data < 0.6)]),
                                'poor': len(habitability_data[habitability_data < 0.4])
                            }
                        }
                    
                    # Microorganism data
                    if 'suitable microorganisms' in df.columns:
                        microbes = df['suitable microorganisms'].dropna()
                        all_microbes = []
                        for microbe_list in microbes:
                            if isinstance(microbe_list, str):
                                all_microbes.extend([m.strip() for m in microbe_list.split(',')])
                        
                        if all_microbes:
                            microbe_counts = pd.Series(all_microbes).value_counts()
                            insights['microorganism_data'] = {
                                'total_unique_microbes': len(microbe_counts),
                                'most_common_microbes': microbe_counts.head(10).to_dict(),
                                'microbe_frequency': microbe_counts.to_dict()
                            }
                            
            except Exception as e:
                print(f"[CHAT] Error extracting insights: {e}")
                insights = None
            
            # Get enhanced response from trained chatbot
            chatbot_response = get_chatbot_response(dashboard, question, insights)
            
            print(f"[CHAT] Enhanced response: {chatbot_response['response']}")
            print(f"[CHAT] Confidence: {chatbot_response['confidence']}, Source: {chatbot_response['source']}")
            
            return jsonify({
                "response": chatbot_response['response'],
                "confidence": chatbot_response['confidence'],
                "source": chatbot_response['source']
            })
            
        except ImportError:
            print("[CHAT] Enhanced chatbot not available, falling back to basic response")
            # Fallback to basic response if enhanced chatbot is not available
            return _get_basic_chat_response(dashboard, question)
            
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        return jsonify({"response": "I'm sorry, I'm having trouble processing your question right now. Please try again.", "error": str(e)}), 500

def _get_basic_chat_response(dashboard, question):
    """Fallback basic chat response"""
    try:
        # ✅ Run PPO-based viability analysis
        result = analyze_data(dashboard)
        if "error" in result:
            return jsonify({"response": "Sorry, I couldn't analyze the data right now."})

        ratio = result["viability_ratio"]
        confidence = result["last_prediction_confidence"]

        # ✅ Load microbe model and encoder
        model_path = f"backend/model/{dashboard}_microbe_model.pkl"
        encoder_path = f"backend/model/{dashboard}_mlb.pkl"
        microbes = []

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            model = joblib.load(model_path)
            mlb = joblib.load(encoder_path)

            df = load_dashboard_data(dashboard, numeric_only=False)
            feature_cols = [
                "Min Temperature (°C)", "Max Temperature (°C)", "Humidity (%)",
                "Radiation (mSv/day)", "Perchlorates", "Sulfates",
                "Ice Depth (m)", "Habitability Score"
            ] if dashboard == "mars" else [
                "Min Temperature (°C)", "Max Temperature (°C)", "Radiation (mSv/day)",
                "Water Ice Present", "Ilmenite Content", "Regolith Depth (m)",
                "Sunlight Hours", "Habitability Score"
            ]

            latest = df.dropna(subset=feature_cols).iloc[-1:]
            X = pd.get_dummies(latest[feature_cols])
            X = X.reindex(columns=model.estimators_[0].feature_importances_.shape[0], fill_value=0)

            pred = model.predict(X)
            microbes = mlb.inverse_transform(pred)[0]

        # ✅ Build response
        if "microorganism" in question.lower() or "viable" in question.lower():
            if ratio > 0.7:
                message = f"Yes, microorganisms are likely viable on {dashboard.capitalize()} today. Viability ratio: {ratio}, confidence: {confidence}."
            elif ratio > 0.4:
                message = f"Microorganism viability on {dashboard.capitalize()} is moderate. Ratio: {ratio}, confidence: {confidence}."
            else:
                message = f"Conditions on {dashboard.capitalize()} are not favorable for microorganisms. Ratio: {ratio}, confidence: {confidence}."

            if microbes:
                message += f" Predicted viable microbes: {', '.join(microbes)}."
        else:
            message = f"{dashboard.capitalize()} biosphere analysis complete. Viability ratio: {ratio}, confidence: {confidence}."
            if microbes:
                message += f" Predicted viable microbes: {', '.join(microbes)}."

        return jsonify({"response": message, "confidence": confidence, "source": "basic_analysis"})

    except Exception as e:
        print(f"[BASIC CHAT ERROR] {e}")
        return jsonify({"response": "I'm having trouble analyzing the data. Please try again later.", "error": str(e)}), 500

# 🕒 Last AI update
@app.route('/last-update', methods=['GET'])
def last_update():
    try:
        dashboard = request.args.get("dashboard", "moon")  # ✅ Allow frontend to pass dashboard
        update_time = get_last_update(dashboard)
        print(f"[LAST UPDATE] {dashboard}: {update_time}")
        return jsonify({"last_update": update_time})
    except Exception as e:
        print(f"[LAST UPDATE ERROR] {e}")
        return jsonify({"last_update": "Unavailable", "error": str(e)}), 500

# 🚀 Launch backend + frontend
if __name__ == '__main__':
    threading.Thread(target=start_frontend).start()
    app.run(debug=True)
