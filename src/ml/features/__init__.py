"""
ML Feature Engineering Package

Transforms raw telemetry data into ML-ready feature matrices.

Features:
- FeatureBuilder: Build feature matrices for ML models

Usage:
    from src.ml.features import FeatureBuilder

    builder = FeatureBuilder(session_id=1)

    # Build features for different models
    tyre_features = builder.build_tyre_wear_features(driver_index=0)
    lap_features = builder.build_lap_time_features(driver_index=0)

    # Returns: {'X': feature_matrix, 'y': target, 'feature_names': list}
"""

from .feature_builder import FeatureBuilder

__all__ = ['FeatureBuilder']
