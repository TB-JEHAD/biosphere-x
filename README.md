# biosphere-x
# 🌍 Biosphere-X: Mars & Moon Biosphere Analysis Platform

A comprehensive AI-powered dashboard for analyzing biosphere data from Mars and Moon environments, featuring real-time data visualization, machine learning predictions, and an intelligent chatbot assistant trained on actual biosphere datasets.

## 🚀 Features

### 📊 **Interactive Dashboards**
- **Mars Biosphere Dashboard** - Complete environmental analysis of Martian conditions
- **Moon Biosphere Dashboard** - Comprehensive lunar environment monitoring
- **Real-time 3D Visualizations** - Immersive space environments with rotating planets
- **Dynamic Charts** - Temperature, pressure, radiation, humidity, and atmospheric data

### 🤖 **AI-Powered Analysis**
- **PPO-based Viability Analysis** - Machine learning models for habitability assessment
- **Microorganism Prediction** - AI models trained to predict suitable microorganisms
- **Enhanced Chatbot Assistant** - Intelligent Q&A system trained on actual biosphere data
- **Real-time Insights** - Live data analysis with confidence scoring

### 📈 **Data Visualization**
- **Multi-metric Charts** - Temperature, pressure, wind speed, CO2, humidity tracking
- **Prediction Overlays** - Actual vs predicted data visualization
- **Interactive Controls** - Upload, analyze, and retrain models on demand
- **Responsive Design** - Optimized for desktop and mobile viewing

### 🧠 **Smart Chatbot**
- **Data-Trained Intelligence** - Trained on 2,536 Mars records and 3,312 Moon records
- **Context-Aware Responses** - Understands questions about temperature, pressure, radiation, habitability
- **Multi-Source Analysis** - Combines knowledge base with live data insights
- **Confidence Scoring** - Provides reliability indicators for responses

## 📁 Project Structure

```
biosphere-x/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── analyze.py            # PPO-based viability analysis
│   ├── biosphere_env.py      # Reinforcement learning environment
│   ├── chatbot_trainer.py    # AI chatbot training system
│   ├── chatbot_responder.py  # Enhanced chatbot response engine
│   ├── train.py              # PPO model training
│   ├── train_microbe_model.py # Microorganism prediction training
│   ├── utils.py              # Data loading and utility functions
│   ├── scheduler.py          # Background task scheduling
│   └── model/                # Trained AI models and knowledge bases
│       ├── mars_model.zip
│       ├── mars_microbe_model.pkl
│       ├── mars_chatbot_kb.json
│       ├── moon_model.zip
│       ├── moon_microbe_model.pkl
│       └── moon_chatbot_kb.json
├── frontend/
│   ├── mars.html            # Mars dashboard interface
│   ├── moon.html            # Moon dashboard interface
│   ├── css/                 # Custom styling
│   └── js/                  # Frontend JavaScript
├── data/
│   ├── mars_data.xlsx       # Mars biosphere dataset (2,536 records)
│   └── moon_data.xlsx       # Moon biosphere dataset (3,312 records)
└── requirements.txt         # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Modern web browser with JavaScript enabled

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/TB-JEHAD/biosphere-x.git
   cd biosphere-x
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python backend/app.py
   ```

4. **Access the dashboards**
   - Mars Dashboard: `http://localhost:8000/mars.html`
   - Moon Dashboard: `http://localhost:8000/moon.html`
   - API Endpoints: `http://localhost:5000/`

## 📊 Dataset Information

### Mars Biosphere Data (2,536 records)
- **Time Period**: 2000-2025
- **Locations**: Gale Crater, Elysium Planitia, Utopia Planitia, Meridiani Planum, Jezero Crater, Polar Regions, Valles Marineris
- **Metrics**: Temperature, atmospheric pressure, humidity, radiation, CO2 levels, wind speed, habitability scores
- **Microorganisms**: 2 identified species (Psychrobacter cryohalolentis, Chroococcidiopsis)

### Moon Biosphere Data (3,312 records)
- **Time Period**: 2009-2025
- **Locations**: Apollo landing sites, South Pole-Aitken Basin, Shackleton Crater, Oceanus Procellarum, Tycho Crater, Mare Tranquillitatis, Peary Crater
- **Metrics**: Temperature, pressure, radiation, sunlight hours, water ice presence, regolith depth, habitability scores

## 🎯 Usage Guide

### Dashboard Navigation
1. **Data Upload**: Upload new biosphere data files (.csv/.xlsx)
2. **Analysis**: Run AI-powered viability analysis
3. **Model Training**: Retrain machine learning models
4. **Chart Viewing**: Monitor real-time environmental metrics
5. **Chat Interface**: Ask questions about biosphere conditions

### Sample Questions for AI Assistant

**Mars Questions:**
- "What is the temperature on Mars?"
- "What microorganisms can survive on Mars?"
- "What is the atmospheric pressure on Mars?"

**Moon Questions:**
- "What is the temperature on the Moon?"
- "How much sunlight does the Moon get?"
- "What are the radiation levels on the Moon?"

## 🔧 API Endpoints

### Core Endpoints
- `POST /upload` - Upload biosphere data files
- `POST /analyze` - Run viability analysis
- `POST /retrain` - Retrain AI models
- `GET /get-chart-data` - Retrieve chart data
- `POST /chat-response` - Get AI assistant responses
- `GET /last-update` - Get last model update time

### Chatbot Training
- `POST /train-chatbot` - Train enhanced chatbot models

## 🧬 AI Models

### PPO (Proximal Policy Optimization)
- **Purpose**: Habitability viability analysis
- **Training**: 10,000 timesteps per dashboard
- **Output**: Viability ratios and confidence scores

### Random Forest Classifier
- **Purpose**: Microorganism prediction
- **Features**: Environmental conditions and habitability scores
- **Output**: Suitable microorganism predictions

### TF-IDF Vectorizer + Cosine Similarity
- **Purpose**: Intelligent question matching
- **Training**: Knowledge base from actual biosphere data
- **Output**: Context-aware responses with confidence scoring

## 📈 Key Metrics

### Mars Environmental Conditions
- **Temperature**: -149.6°C to -40.8°C (avg: -86.9°C)
- **Atmospheric Pressure**: 668-1020 Pa (avg: 775 Pa)
- **Habitability Score**: 0.67 average (146 excellent locations)
- **Suitable Microorganisms**: 2 identified species

### Moon Environmental Conditions
- **Temperature**: -270.1°C to 173.5°C (avg: -65.5°C)
- **Habitability Score**: 0.43 average (1498 poor locations)
- **Radiation Levels**: Variable across lunar surface
- **Sunlight Hours**: Extensive variations by location

## 🔬 Technical Specifications

### Backend Technologies
- **Flask**: Web framework
- **Stable-Baselines3**: PPO reinforcement learning
- **Scikit-learn**: Machine learning models
- **Pandas**: Data processing
- **NumPy**: Numerical computations
- **Three.js**: 3D visualizations

### Frontend Technologies
- **HTML5/CSS3**: Modern web interface
- **JavaScript**: Interactive functionality
- **Chart.js**: Data visualization
- **Three.js**: 3D space environments
- **Font Awesome**: Icons and UI elements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- NASA for providing biosphere data
- OpenAI for AI research inspiration
- The open-source community for excellent libraries and tools

## 📞 Support

For support, questions, or contributions, please open an issue in the repository or contact the development team.
Number: 01686349990

---

**Biosphere-X** - Exploring the frontiers of space habitation through AI-powered analysis 🚀
