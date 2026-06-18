"""
Data loader for Forex currency pair data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class ForexDataLoader:
    """Load and preprocess forex data"""
    
    def __init__(self, pair='EURUSD=X', period='1y'):
        """
        Initialize data loader
        
        Args:
            pair: Currency pair (default: EUR/USD)
            period: Data period (1mo, 3mo, 1y, etc.)
        """
        self.pair = pair
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        self.original_data = None
        
    def load_data(self):
        """Load forex data from yfinance"""
        try:
            import yfinance as yf
            
            print(f"📥 Loading data for {self.pair}...")
            df = yf.download(self.pair, period=self.period, progress=False)
            
            if df.empty:
                print(f"⚠️  No data found for {self.pair}, generating synthetic data...")
                self.data = self._generate_synthetic_data()
            else:
                self.data = df[['Close']].reset_index()
                self.data.columns = ['Date', 'Close']
                print(f"✅ Loaded {len(self.data)} records")
                
            return self.data
            
        except Exception as e:
            print(f"⚠️  Error loading from yfinance: {e}")
            print("Generating synthetic data instead...")
            self.data = self._generate_synthetic_data()
            return self.data
    
    def _generate_synthetic_data(self):
        """Generate synthetic forex data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=365)
        
        # Generate realistic forex price movement
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.005, 365)
        prices = 1.08 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices
        })
        print(f"✅ Generated {len(df)} synthetic records")
        return df
    
    def preprocess(self, lookback_days=60):
        """
        Preprocess data for model training
        
        Args:
            lookback_days: Number of days to use for prediction
            
        Returns:
            X_train, y_train: Training data and labels
        """
        if self.data is None:
            self.load_data()
        
        # Store original data
        self.original_data = self.data['Close'].values.reshape(-1, 1)
        
        # Scale data
        self.scaled_data = self.scaler.fit_transform(self.original_data)
        
        # Create training sequences
        X, y = [], []
        
        for i in range(len(self.scaled_data) - lookback_days):
            X.append(self.scaled_data[i:(i + lookback_days), 0])
            y.append(self.scaled_data[i + lookback_days, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"✅ Preprocessed data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y, self.scaler
    
    def get_latest_sequence(self, lookback_days=60):
        """Get the latest sequence for prediction"""
        if self.scaled_data is None:
            self.load_data()
            self.original_data = self.data['Close'].values.reshape(-1, 1)
            self.scaled_data = self.scaler.fit_transform(self.original_data)
        
        sequence = self.scaled_data[-lookback_days:, 0]
        return sequence.reshape((1, lookback_days, 1))
    
    def inverse_transform(self, scaled_value):
        """Convert scaled value back to original price"""
        return self.scaler.inverse_transform(scaled_value)
