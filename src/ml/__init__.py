"""
F1 Telemetry Machine Learning Module

ML-powered predictive models for strategic decision-making.

ML Models:
- TyreWearModel: Predict tyre wear and degradation
- LapTimeModel: Forecast lap times with degradation
- PitStopOptimizer: Optimize pit stop timing
- RaceOutcomeModel: Predict win/podium/points probabilities
- StrategyRecommender: Comprehensive race strategy recommendations

Features:
- FeatureBuilder: Transform telemetry into ML features

Usage:
    from src.ml.models import TyreWearModel, LapTimeModel, StrategyRecommender
    from src.ml.features import FeatureBuilder

    # Train models
    tyre_model = TyreWearModel()
    tyre_model.train(session_ids=[1, 2, 3])

    # Make predictions
    wear_forecast = tyre_model.predict_wear(current_state)

    # Get strategy recommendations
    recommender = StrategyRecommender()
    strategy = recommender.recommend_strategy(current_state)
"""

from .models import (
    TyreWearModel,
    LapTimeModel,
    PitStopOptimizer,
    RaceOutcomeModel,
    StrategyRecommender
)

from .features import FeatureBuilder

__all__ = [
    'TyreWearModel',
    'LapTimeModel',
    'PitStopOptimizer',
    'RaceOutcomeModel',
    'StrategyRecommender',
    'FeatureBuilder'
]
