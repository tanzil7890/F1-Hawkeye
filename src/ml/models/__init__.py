"""
ML Models Package

Predictive models for F1 telemetry analysis.

Models:
- TyreWearModel: Tyre wear and degradation prediction
- LapTimeModel: Lap time forecasting with degradation
- PitStopOptimizer: Pit stop timing optimization
- RaceOutcomeModel: Win/podium/points probability prediction
- StrategyRecommender: Comprehensive race strategy recommendations

Usage:
    from src.ml.models import TyreWearModel, LapTimeModel, StrategyRecommender

    # Individual models
    tyre_model = TyreWearModel()
    tyre_model.train(session_ids=[1, 2, 3])
    forecast = tyre_model.predict_wear(current_state)

    # Integrated strategy
    recommender = StrategyRecommender()
    strategy = recommender.recommend_strategy(current_state)
"""

from .tyre_wear_model import TyreWearModel
from .lap_time_model import LapTimeModel
from .pit_stop_optimizer import PitStopOptimizer
from .race_outcome_model import RaceOutcomeModel
from .strategy_model import StrategyRecommender

__all__ = [
    'TyreWearModel',
    'LapTimeModel',
    'PitStopOptimizer',
    'RaceOutcomeModel',
    'StrategyRecommender'
]
