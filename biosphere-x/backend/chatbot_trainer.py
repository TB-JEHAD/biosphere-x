import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class BiosphereChatbotTrainer:
    def __init__(self):
        self.data_dir = "data"
        self.model_dir = "backend/model"
        self.knowledge_base = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def load_and_process_data(self, dashboard):
        """Load and process the biosphere data for training"""
        file_path = os.path.join(self.data_dir, f"{dashboard}_data.xlsx")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        df = pd.read_excel(file_path)
        
        # Convert all columns to lowercase for consistency
        df.columns = [col.lower().strip() for col in df.columns]
        
        return df
    
    def extract_key_insights(self, df, dashboard):
        """Extract key insights and patterns from the data"""
        insights = {
            'dashboard': dashboard,
            'data_summary': {},
            'environmental_patterns': {},
            'habitability_analysis': {},
            'microorganism_data': {},
            'temporal_patterns': {},
            'statistical_summary': {}
        }
        
        # Basic data summary
        insights['data_summary'] = {
            'total_records': len(df),
            'date_range': {
                'start': str(df['earth date'].min()) if 'earth date' in df.columns else 'N/A',
                'end': str(df['earth date'].max()) if 'earth date' in df.columns else 'N/A'
            },
            'locations': df['location name'].unique().tolist() if 'location name' in df.columns else [],
            'sites': df['site id'].unique().tolist() if 'site id' in df.columns else []
        }
        
        # Environmental patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['site id', 'latitude', 'longitude']:  # Skip ID and coordinate columns
                insights['environmental_patterns'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'trend': self._calculate_trend(df[col])
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
        
        # Temporal patterns (if date column exists)
        if 'earth date' in df.columns:
            df['earth date'] = pd.to_datetime(df['earth date'], errors='coerce')
            temporal_data = df.dropna(subset=['earth date'])
            
            if not temporal_data.empty:
                # Convert Period objects to strings for JSON serialization
                records_per_month = temporal_data.groupby(temporal_data['earth date'].dt.to_period('M')).size()
                records_per_month_dict = {str(k): int(v) for k, v in records_per_month.items()}
                
                insights['temporal_patterns'] = {
                    'data_span_days': int((temporal_data['earth date'].max() - temporal_data['earth date'].min()).days),
                    'records_per_month': records_per_month_dict,
                    'seasonal_patterns': self._analyze_seasonal_patterns(temporal_data)
                }
        
        return insights
    
    def _calculate_trend(self, series):
        """Calculate trend direction for a time series"""
        if len(series) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return "insufficient_data"
            
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calculate slope
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _analyze_seasonal_patterns(self, df):
        """Analyze seasonal patterns in the data"""
        patterns = {}
        
        # Group by month
        monthly_data = df.groupby(df['earth date'].dt.month)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['site id', 'latitude', 'longitude']:
                monthly_avg = monthly_data[col].mean()
                patterns[col] = {
                    'monthly_averages': monthly_avg.to_dict(),
                    'peak_month': int(monthly_avg.idxmax()),
                    'low_month': int(monthly_avg.idxmin())
                }
        
        return patterns
    
    def generate_knowledge_base(self, dashboard):
        """Generate a comprehensive knowledge base for the chatbot"""
        print(f"[CHATBOT TRAINER] Generating knowledge base for {dashboard}...")
        
        df = self.load_and_process_data(dashboard)
        insights = self.extract_key_insights(df, dashboard)
        
        # Create Q&A pairs based on the insights
        qa_pairs = self._generate_qa_pairs(insights, df)
        
        # Store in knowledge base
        self.knowledge_base[dashboard] = {
            'insights': insights,
            'qa_pairs': qa_pairs,
            'data_sample': df.head(10).to_dict('records'),  # Sample for reference
            'last_updated': datetime.now().isoformat()
        }
        
        return self.knowledge_base[dashboard]
    
    def _generate_qa_pairs(self, insights, df):
        """Generate question-answer pairs from the insights"""
        qa_pairs = []
        dashboard = insights['dashboard']
        
        # General information questions
        qa_pairs.extend([
            {
                'question': f'what is the {dashboard} biosphere data about?',
                'answer': f'The {dashboard} biosphere data contains {insights["data_summary"]["total_records"]} records of environmental conditions and habitability measurements. It covers data from {insights["data_summary"]["date_range"]["start"]} to {insights["data_summary"]["date_range"]["end"]} across multiple locations.'
            },
            {
                'question': f'how many records are in the {dashboard} dataset?',
                'answer': f'The {dashboard} dataset contains {insights["data_summary"]["total_records"]} records of biosphere measurements.'
            }
        ])
        
        # Temperature questions
        if 'min temperature (°c)' in insights['environmental_patterns']:
            temp_data = insights['environmental_patterns']['min temperature (°c)']
            qa_pairs.extend([
                {
                    'question': f'what is the temperature range on {dashboard}?',
                    'answer': f'On {dashboard}, temperatures range from {temp_data["min"]:.1f}°C to {temp_data["max"]:.1f}°C, with an average of {temp_data["mean"]:.1f}°C. The trend is {temp_data["trend"]}.'
                },
                {
                    'question': f'what is the average temperature on {dashboard}?',
                    'answer': f'The average temperature on {dashboard} is {temp_data["mean"]:.1f}°C, with a standard deviation of {temp_data["std"]:.1f}°C.'
                }
            ])
        
        # Pressure questions
        if 'pressure (pa)' in insights['environmental_patterns']:
            pressure_data = insights['environmental_patterns']['pressure (pa)']
            qa_pairs.extend([
                {
                    'question': f'what is the atmospheric pressure on {dashboard}?',
                    'answer': f'The atmospheric pressure on {dashboard} ranges from {pressure_data["min"]:.2e} Pa to {pressure_data["max"]:.2e} Pa, with an average of {pressure_data["mean"]:.2e} Pa.'
                }
            ])
        
        # Radiation questions
        if 'radiation (msv/day)' in insights['environmental_patterns']:
            radiation_data = insights['environmental_patterns']['radiation (msv/day)']
            qa_pairs.extend([
                {
                    'question': f'what are the radiation levels on {dashboard}?',
                    'answer': f'Radiation levels on {dashboard} average {radiation_data["mean"]:.2f} mSv/day, ranging from {radiation_data["min"]:.2f} to {radiation_data["max"]:.2f} mSv/day.'
                }
            ])
        
        # Habitability questions
        if insights['habitability_analysis']:
            habit_data = insights['habitability_analysis']
            qa_pairs.extend([
                {
                    'question': f'how habitable is {dashboard}?',
                    'answer': f'{dashboard.capitalize()} has an average habitability score of {habit_data["average_score"]:.2f}. {habit_data["score_distribution"]["excellent"]} locations are excellent, {habit_data["score_distribution"]["good"]} are good, {habit_data["score_distribution"]["moderate"]} are moderate, and {habit_data["score_distribution"]["poor"]} are poor for habitability.'
                },
                {
                    'question': f'what is the habitability score of {dashboard}?',
                    'answer': f'The habitability score on {dashboard} ranges from {habit_data["score_range"]["min"]:.2f} to {habit_data["score_range"]["max"]:.2f}, with an average of {habit_data["average_score"]:.2f}.'
                }
            ])
        
        # Microorganism questions
        if insights['microorganism_data']:
            microbe_data = insights['microorganism_data']
            top_microbes = list(microbe_data['most_common_microbes'].keys())[:5]
            qa_pairs.extend([
                {
                    'question': f'what microorganisms can survive on {dashboard}?',
                    'answer': f'The data identifies {microbe_data["total_unique_microbes"]} different microorganisms that can survive on {dashboard}. The most common ones include: {", ".join(top_microbes)}.'
                },
                {
                    'question': f'how many microorganisms are suitable for {dashboard}?',
                    'answer': f'According to the data, {microbe_data["total_unique_microbes"]} different microorganisms are suitable for survival on {dashboard}.'
                }
            ])
        
        # Dashboard-specific questions
        if dashboard == 'mars':
            if 'humidity (%)' in insights['environmental_patterns']:
                humidity_data = insights['environmental_patterns']['humidity (%)']
                qa_pairs.append({
                    'question': 'what is the humidity on mars?',
                    'answer': f'Humidity on Mars averages {humidity_data["mean"]:.1f}%, ranging from {humidity_data["min"]:.1f}% to {humidity_data["max"]:.1f}%.'
                })
            
            if 'atmospheric co2 (%)' in insights['environmental_patterns']:
                co2_data = insights['environmental_patterns']['atmospheric co2 (%)']
                qa_pairs.append({
                    'question': 'what is the co2 level on mars?',
                    'answer': f'Atmospheric CO2 on Mars averages {co2_data["mean"]:.1f}%, ranging from {co2_data["min"]:.1f}% to {co2_data["max"]:.1f}%.'
                })
        
        elif dashboard == 'moon':
            if 'sunlight hours' in insights['environmental_patterns']:
                sunlight_data = insights['environmental_patterns']['sunlight hours']
                qa_pairs.extend([
                    {
                        'question': 'how much sunlight does the moon get?',
                        'answer': f'The Moon receives an average of {sunlight_data["mean"]:.1f} sunlight hours, ranging from {sunlight_data["min"]:.1f} to {sunlight_data["max"]:.1f} hours.'
                    },
                    {
                        'question': 'what are the sunlight conditions on the moon?',
                        'answer': f'Moon sunlight conditions average {sunlight_data["mean"]:.1f} hours, with a trend that is {sunlight_data["trend"]}.'
                    }
                ])
            
            if 'water ice present' in insights['environmental_patterns']:
                ice_data = insights['environmental_patterns']['water ice present']
                qa_pairs.append({
                    'question': 'is there water ice on the moon?',
                    'answer': f'Water ice presence on the Moon shows {ice_data["mean"]:.1f} average presence, ranging from {ice_data["min"]:.1f} to {ice_data["max"]:.1f}.'
                })
        
        return qa_pairs
    
    def train_vectorizer(self, dashboard):
        """Train TF-IDF vectorizer on the Q&A pairs"""
        if dashboard not in self.knowledge_base:
            self.generate_knowledge_base(dashboard)
        
        qa_pairs = self.knowledge_base[dashboard]['qa_pairs']
        questions = [qa['question'] for qa in qa_pairs]
        
        # Fit the vectorizer
        self.vectorizer.fit(questions)
        
        # Vectorize the questions
        question_vectors = self.vectorizer.transform(questions)
        
        return question_vectors, questions
    
    def save_trained_model(self, dashboard):
        """Save the trained chatbot model and knowledge base"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save knowledge base
        kb_path = os.path.join(self.model_dir, f"{dashboard}_chatbot_kb.json")
        with open(kb_path, 'w') as f:
            json.dump(self.knowledge_base[dashboard], f, indent=2, default=str)
        
        # Train and save vectorizer
        question_vectors, questions = self.train_vectorizer(dashboard)
        vectorizer_path = os.path.join(self.model_dir, f"{dashboard}_chatbot_vectorizer.pkl")
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save question vectors and questions
        vectors_path = os.path.join(self.model_dir, f"{dashboard}_chatbot_vectors.pkl")
        joblib.dump(question_vectors, vectors_path)
        
        questions_path = os.path.join(self.model_dir, f"{dashboard}_chatbot_questions.pkl")
        joblib.dump(questions, questions_path)
        
        print(f"[CHATBOT TRAINER] Model saved for {dashboard}:")
        print(f"  - Knowledge base: {kb_path}")
        print(f"  - Vectorizer: {vectorizer_path}")
        print(f"  - Question vectors: {vectors_path}")
        print(f"  - Questions: {questions_path}")
    
    def train_all_dashboards(self):
        """Train chatbot models for all available dashboards"""
        dashboards = ['mars', 'moon']
        
        for dashboard in dashboards:
            try:
                print(f"\n[CHATBOT TRAINER] Training {dashboard} chatbot...")
                self.generate_knowledge_base(dashboard)
                self.save_trained_model(dashboard)
                print(f"[CHATBOT TRAINER] ✅ {dashboard.capitalize()} chatbot training completed!")
            except Exception as e:
                print(f"[CHATBOT TRAINER] ❌ Error training {dashboard} chatbot: {e}")

def train_chatbot_models():
    """Main function to train all chatbot models"""
    trainer = BiosphereChatbotTrainer()
    trainer.train_all_dashboards()
    return trainer

if __name__ == "__main__":
    train_chatbot_models()
