"""
LSTM Model for Forex Price Prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ForexLSTMModel:
    """LSTM model for forex prediction"""
    
    def __init__(self, lookback_days=60, model_path='models/forex_model.h5'):
        """
        Initialize LSTM model
        
        Args:
            lookback_days: Number of days for LSTM lookback
            model_path: Path to save/load model
        """
        self.lookback_days = lookback_days
        self.model_path = model_path
        self.model = None
        self.history = None
        self._create_model()
        
    def _create_model(self):
        """Create LSTM model architecture"""
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.lookback_days, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(25, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Model created successfully")
        self.model.summary()
        
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        print(f"🎓 Training model for {epochs} epochs...")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=True
        )
        
        print("✅ Training completed")
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X, verbose=0)
    
    def save(self):
        """Save model to disk"""
        os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
        self.model.save(self.model_path)
        print(f"💾 Model saved to {self.model_path}")
    
    def load(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"📂 Model loaded from {self.model_path}")
            return True
        return False
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            loss, mae
        """
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, mae
