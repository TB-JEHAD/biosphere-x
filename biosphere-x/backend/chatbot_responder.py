import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime

class BiosphereChatbotResponder:
    def __init__(self):
        self.model_dir = "backend/model"
        self.knowledge_bases = {}
        self.vectorizers = {}
        self.question_vectors = {}
        self.questions = {}
        
    def load_model(self, dashboard):
        """Load the trained chatbot model for a specific dashboard"""
        if dashboard in self.knowledge_bases:
            return True
            
        try:
            # Load knowledge base
            kb_path = os.path.join(self.model_dir, f"{dashboard}_chatbot_kb.json")
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    self.knowledge_bases[dashboard] = json.load(f)
            else:
                return False
            
            # Load vectorizer
            vectorizer_path = os.path.join(self.model_dir, f"{dashboard}_chatbot_vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                self.vectorizers[dashboard] = joblib.load(vectorizer_path)
            else:
                return False
            
            # Load question vectors
            vectors_path = os.path.join(self.model_dir, f"{dashboard}_chatbot_vectors.pkl")
            if os.path.exists(vectors_path):
                self.question_vectors[dashboard] = joblib.load(vectors_path)
            else:
                return False
            
            # Load questions
            questions_path = os.path.join(self.model_dir, f"{dashboard}_chatbot_questions.pkl")
            if os.path.exists(questions_path):
                self.questions[dashboard] = joblib.load(questions_path)
            else:
                return False
            
            print(f"[CHATBOT RESPONDER] Loaded model for {dashboard}")
            return True
            
        except Exception as e:
            print(f"[CHATBOT RESPONDER] Error loading model for {dashboard}: {e}")
            return False
    
    def find_best_answer(self, dashboard, question):
        """Find the best answer for a question using similarity matching"""
        if not self.load_model(dashboard):
            return None
        
        # Clean and preprocess the question
        clean_question = self._preprocess_question(question)
        
        # Vectorize the question
        question_vector = self.vectorizers[dashboard].transform([clean_question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, self.question_vectors[dashboard]).flatten()
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Only return if similarity is above threshold
        if best_similarity > 0.1:  # Low threshold for flexibility
            qa_pairs = self.knowledge_bases[dashboard]['qa_pairs']
            return {
                'answer': qa_pairs[best_match_idx]['answer'],
                'confidence': float(best_similarity),
                'matched_question': qa_pairs[best_match_idx]['question']
            }
        
        return None
    
    def _preprocess_question(self, question):
        """Preprocess the question for better matching"""
        # Convert to lowercase
        question = question.lower()
        
        # Remove special characters but keep spaces
        question = re.sub(r'[^\w\s]', '', question)
        
        # Remove extra whitespace
        question = ' '.join(question.split())
        
        return question
    
    def generate_contextual_response(self, dashboard, question, insights=None):
        """Generate a contextual response using both knowledge base and live data"""
        # Try to find a direct answer from knowledge base
        kb_answer = self.find_best_answer(dashboard, question)
        
        if kb_answer and kb_answer['confidence'] > 0.3:
            # High confidence match from knowledge base
            response = kb_answer['answer']
            
            # Add real-time data if available
            if insights:
                response = self._enhance_with_live_data(response, insights, dashboard)
            
            return {
                'response': response,
                'confidence': kb_answer['confidence'],
                'source': 'knowledge_base'
            }
        
        # Try to generate a response based on live data analysis
        if insights:
            live_response = self._generate_live_data_response(question, insights, dashboard)
            if live_response:
                return {
                    'response': live_response,
                    'confidence': 0.5,
                    'source': 'live_analysis'
                }
        
        # Fallback to general response
        return self._generate_fallback_response(dashboard, question)
    
    def _enhance_with_live_data(self, base_response, insights, dashboard):
        """Enhance knowledge base response with current data insights"""
        enhanced_response = base_response
        
        # Add current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        enhanced_response += f" (Data current as of {current_time})"
        
        # Add any recent trends or alerts
        if 'environmental_patterns' in insights:
            trends = []
            for metric, data in insights['environmental_patterns'].items():
                if data.get('trend') in ['increasing', 'decreasing']:
                    trends.append(f"{metric.replace('_', ' ').title()} is {data['trend']}")
            
            if trends:
                enhanced_response += f" Current trends: {', '.join(trends[:3])}."
        
        return enhanced_response
    
    def _generate_live_data_response(self, question, insights, dashboard):
        """Generate a response based on live data analysis"""
        question_lower = question.lower()
        
        # Temperature questions
        if any(word in question_lower for word in ['temperature', 'temp', 'hot', 'cold', 'warm']):
            if 'min temperature (°c)' in insights.get('environmental_patterns', {}):
                temp_data = insights['environmental_patterns']['min temperature (°c)']
                return f"Current temperature data shows an average of {temp_data['mean']:.1f}°C, ranging from {temp_data['min']:.1f}°C to {temp_data['max']:.1f}°C. The trend is {temp_data['trend']}."
        
        # Pressure questions
        if any(word in question_lower for word in ['pressure', 'atmospheric', 'atmosphere']):
            if 'pressure (pa)' in insights.get('environmental_patterns', {}):
                pressure_data = insights['environmental_patterns']['pressure (pa)']
                return f"Atmospheric pressure averages {pressure_data['mean']:.2e} Pa, with values ranging from {pressure_data['min']:.2e} to {pressure_data['max']:.2e} Pa."
        
        # Radiation questions
        if any(word in question_lower for word in ['radiation', 'rad', 'exposure']):
            if 'radiation (msv/day)' in insights.get('environmental_patterns', {}):
                rad_data = insights['environmental_patterns']['radiation (msv/day)']
                return f"Radiation levels average {rad_data['mean']:.2f} mSv/day, ranging from {rad_data['min']:.2f} to {rad_data['max']:.2f} mSv/day."
        
        # Habitability questions
        if any(word in question_lower for word in ['habitable', 'habitability', 'viable', 'suitable']):
            if insights.get('habitability_analysis'):
                habit_data = insights['habitability_analysis']
                return f"Habitability analysis shows an average score of {habit_data['average_score']:.2f}, with {habit_data['score_distribution']['excellent']} excellent locations and {habit_data['score_distribution']['poor']} poor locations."
        
        # Microorganism questions
        if any(word in question_lower for word in ['microorganism', 'microbe', 'bacteria', 'organism']):
            if insights.get('microorganism_data'):
                microbe_data = insights['microorganism_data']
                top_microbes = list(microbe_data['most_common_microbes'].keys())[:3]
                return f"Data identifies {microbe_data['total_unique_microbes']} microorganisms suitable for {dashboard}, with the most common being: {', '.join(top_microbes)}."
        
        return None
    
    def _generate_fallback_response(self, dashboard, question):
        """Generate a fallback response when no specific answer is found"""
        fallback_responses = {
            'mars': [
                f"I'm analyzing Mars biosphere data to provide accurate information about environmental conditions, habitability, and potential for life. Could you ask about specific aspects like temperature, pressure, radiation, or habitability scores?",
                f"Mars biosphere data includes measurements of temperature, atmospheric pressure, radiation levels, and habitability assessments. What specific information would you like to know?",
                f"I have access to comprehensive Mars environmental data. You can ask about current conditions, trends, habitability scores, or suitable microorganisms."
            ],
            'moon': [
                f"I'm processing Moon biosphere data covering environmental conditions, radiation exposure, and habitability factors. What specific aspect of lunar conditions would you like to explore?",
                f"Moon biosphere data includes sunlight hours, radiation levels, temperature variations, and habitability assessments. How can I help you understand lunar conditions?",
                f"I have detailed Moon environmental data available. You can ask about sunlight conditions, radiation levels, temperature ranges, or habitability scores."
            ]
        }
        
        import random
        return {
            'response': random.choice(fallback_responses.get(dashboard, fallback_responses['mars'])),
            'confidence': 0.1,
            'source': 'fallback'
        }

def get_chatbot_response(dashboard, question, insights=None):
    """Main function to get chatbot response"""
    responder = BiosphereChatbotResponder()
    return responder.generate_contextual_response(dashboard, question, insights)
