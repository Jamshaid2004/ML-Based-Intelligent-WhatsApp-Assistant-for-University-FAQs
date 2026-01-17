import pandas as pd
from datetime import datetime
import os

class ConversationLogger:
    """Logs WhatsApp conversations"""
    
    def __init__(self, log_file='data/whatsapp_logs.csv'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            self.logs = pd.read_csv(log_file)
        except FileNotFoundError:
            self.logs = pd.DataFrame(columns=[
                'timestamp', 'user_id', 'user_question', 
                'predicted_intent', 'confidence', 'response'
            ])
    
    def log_interaction(self, question, intent, confidence, user_id=None, response=None):
        """Log interaction"""
        new_log = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_question': question,
            'predicted_intent': intent,
            'confidence': confidence,
            'response': response
        }])
        
        self.logs = pd.concat([self.logs, new_log], ignore_index=True)
        self.logs.to_csv(self.log_file, index=False)
    
    def get_analytics(self):
        """Get analytics"""
        if len(self.logs) == 0:
            return {
                'total_interactions': 0,
                'unique_users': 0,
                'top_intents': {}
            }
        
        return {
            'total_interactions': len(self.logs),
            'unique_users': self.logs['user_id'].nunique(),
            'average_confidence': self.logs['confidence'].mean(),
            'top_intents': self.logs['predicted_intent'].value_counts().head(10).to_dict(),
            'daily_volume': self.logs.groupby(
                pd.to_datetime(self.logs['timestamp']).dt.date
            ).size().to_dict()
        }