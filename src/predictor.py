"""
Main predictor class combining data loading, training, and prediction
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_loader import ForexDataLoader
from src.model import ForexLSTMModel
import warnings

warnings.filterwarnings('ignore')


class ForexPredictor:
    """Complete forex prediction pipeline"""
    
    def __init__(self, pair='EURUSD=X', lookback_days=60):
        """
        Initialize predictor
        
        Args:
            pair: Currency pair
            lookback_days: Lookback period for LSTM
        """
        self.pair = pair
        self.lookback_days = lookback_days
        self.data_loader = ForexDataLoader(pair=pair)
        self.model = ForexLSTMModel(lookback_days=lookback_days)
        self.X_train = None
        self.y_train = None
        self.scaler = None
        self.last_prices = None
        
    def train(self, epochs=50, batch_size=32):
        """
        Train the complete pipeline
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Forex AI Predictor for {self.pair}")
        print(f"{'='*60}\n")
        
        # Load and preprocess data
        print("Step 1: Loading data...")
        self.data_loader.load_data()
        
        print("Step 2: Preprocessing data...")
        self.X_train, self.y_train, self.scaler = self.data_loader.preprocess(
            lookback_days=self.lookback_days
        )
        
        # Split data
        split = int(len(self.X_train) * 0.8)
        X_train_split = self.X_train[:split]
        y_train_split = self.y_train[:split]
        X_test = self.X_train[split:]
        y_test = self.y_train[split:]
        
        # Train model
        print("Step 3: Training model...")
        self.model.train(
            X_train_split, y_train_split,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate
        print("Step 4: Evaluating model...")
        loss, mae = self.model.evaluate(X_test, y_test)
        
        # Get predictions for metrics
        y_pred = self.model.predict(X_test).flatten()
        
        # Inverse transform for interpretability
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_original = self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        print(f"\n📊 Model Performance:")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE:  {mae_original:.6f}")
        print(f"   R²:   {r2:.4f}")
        
        # Save model
        self.model.save()
        
        return {
            'rmse': rmse,
            'mae': mae_original,
            'r2': r2
        }
    
    def predict(self, days_ahead=30):
        """
        Predict future prices
        
        Args:
            days_ahead: Number of days to predict
            
        Returns:
            Predictions
        """
        print(f"\n🔮 Predicting next {days_ahead} days for {self.pair}...")
        
        # Get latest sequence
        current_sequence = self.data_loader.get_latest_sequence(
            lookback_days=self.lookback_days
        )
        
        predictions = []
        current_sequence_copy = current_sequence.copy()
        
        # Generate predictions
        for i in range(days_ahead):
            next_pred = self.model.predict(current_sequence_copy)
            predictions.append(next_pred[0, 0])
            
            # Update sequence with new prediction
            current_sequence_copy = np.append(
                current_sequence_copy[:, 1:, :],
                next_pred.reshape(1, 1, 1),
                axis=1
            )
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_original = self.scaler.inverse_transform(predictions)
        
        print(f"✅ Predictions generated")
        print(f"\nForecast for next {days_ahead} days:")
        for i, pred in enumerate(predictions_original[:min(10, days_ahead)], 1):
            print(f"   Day {i}: {pred[0]:.6f}")
        if days_ahead > 10:
            print(f"   ... and {days_ahead - 10} more days")
        
        return predictions_original.flatten()
    
    def get_current_price(self):
        """Get current price"""
        if self.data_loader.data is not None:
            return self.data_loader.data['Close'].iloc[-1]
        return None
