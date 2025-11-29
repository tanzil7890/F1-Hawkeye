"""
ML Inference/Prediction Package

Real-time prediction engine for race strategy.

Modules:
- predictor: Real-time prediction engine

Usage:
    from src.ml.inference import RealTimePredictor

    predictor = RealTimePredictor()
    predictor.load_models('models/')

    # Real-time predictions
    predictions = predictor.predict(current_state)
"""

from .predictor import RealTimePredictor

__all__ = ['RealTimePredictor']
