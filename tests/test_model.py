"""Tests for model"""

import pytest
import numpy as np
import os
from src.model import ForexLSTMModel


class TestForexLSTMModel:
    """Test ForexLSTMModel class"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = ForexLSTMModel(lookback_days=60)
        assert model.lookback_days == 60
        assert model.model is not None
    
    def test_model_predict(self):
        """Test model prediction"""
        model = ForexLSTMModel(lookback_days=60)
        
        X = np.random.randn(10, 60, 1)
        predictions = model.predict(X)
        
        assert predictions.shape == (10, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
