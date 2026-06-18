"""Tests for data loader"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import ForexDataLoader


class TestForexDataLoader:
    """Test ForexDataLoader class"""
    
    def test_initialization(self):
        """Test data loader initialization"""
        loader = ForexDataLoader(pair='EURUSD=X', period='1mo')
        assert loader.pair == 'EURUSD=X'
        assert loader.period == '1mo'
    
    def test_load_data(self):
        """Test loading data"""
        loader = ForexDataLoader()
        data = loader.load_data()
        
        assert data is not None
        assert 'Date' in data.columns
        assert 'Close' in data.columns
        assert len(data) > 0
    
    def test_preprocess(self):
        """Test data preprocessing"""
        loader = ForexDataLoader()
        loader.load_data()
        
        X, y, scaler = loader.preprocess(lookback_days=60)
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 60
        assert X.shape[2] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
