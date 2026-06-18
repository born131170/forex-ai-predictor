"""Tests for predictor"""

import pytest
import numpy as np
from src.predictor import ForexPredictor


class TestForexPredictor:
    """Test ForexPredictor class"""
    
    def test_initialization(self):
        """Test predictor initialization"""
        predictor = ForexPredictor(pair='EURUSD=X', lookback_days=60)
        
        assert predictor.pair == 'EURUSD=X'
        assert predictor.lookback_days == 60
        assert predictor.model is not None
    
    def test_get_current_price(self):
        """Test getting current price"""
        predictor = ForexPredictor()
        predictor.data_loader.load_data()
        
        price = predictor.get_current_price()
        assert price is not None
        assert price > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
