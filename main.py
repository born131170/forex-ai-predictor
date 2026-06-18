#!/usr/bin/env python3
"""
Main entry point for Forex AI Predictor
Run this file to train and test the application
"""

import os
import sys
from src.predictor import ForexPredictor
import warnings

warnings.filterwarnings('ignore')


def main():
    """Main function to run the predictor"""
    
    print("\n" + "="*70)
    print("  🚀 FOREX AI PREDICTOR - DEMO APPLICATION 🚀")
    print("="*70 + "\n")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    try:
        # Initialize predictor for EUR/USD
        print("📊 Initializing Forex AI Predictor for EUR/USD...")
        predictor = ForexPredictor(pair='EURUSD=X', lookback_days=60)
        
        # Train model
        print("\n" + "="*70)
        print("STEP 1: TRAINING THE MODEL")
        print("="*70)
        metrics = predictor.train(epochs=50, batch_size=32)
        
        # Get current price
        current_price = predictor.get_current_price()
        print(f"\n💰 Current EUR/USD Price: {current_price:.6f}")
        
        # Make predictions
        print("\n" + "="*70)
        print("STEP 2: MAKING PREDICTIONS")
        print("="*70)
        predictions = predictor.predict(days_ahead=30)
        
        # Display summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n📈 Model Performance Metrics:")
        print(f"   • RMSE: {metrics['rmse']:.6f}")
        print(f"   • MAE:  {metrics['mae']:.6f}")
        print(f"   • R²:   {metrics['r2']:.4f}")
        
        print(f"\n💹 Prediction Summary:")
        print(f"   • Current Price:  {current_price:.6f}")
        print(f"   • Min Predicted:  {predictions.min():.6f}")
        print(f"   • Max Predicted:  {predictions.max():.6f}")
        print(f"   • Avg Predicted:  {predictions.mean():.6f}")
        print(f"   • Day 1 Forecast: {predictions[0]:.6f}")
        print(f"   • Day 30 Forecast: {predictions[-1]:.6f}")
        
        # Test results
        print("\n" + "="*70)
        print("✅ APPLICATION TEST SUCCESSFUL")
        print("="*70)
        print("\n📝 Next Steps:")
        print("   1. Check 'models/' folder for saved model")
        print("   2. Run 'pytest' to run the test suite")
        print("   3. Modify parameters in main.py to experiment")
        print("\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
