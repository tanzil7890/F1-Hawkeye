"""
F1 Telemetry Analysis Module

Transforms raw telemetry into meaningful features for ML models.

Analytics Modules:
- TyreAnalytics: Wear rates, degradation curves, temperature analysis
- FuelAnalytics: Consumption rates, stint planning, fuel-corrected pace
- LapAnalytics: Ideal laps, consistency scores, sector analysis
- PaceAnalytics: Race pace, tyre-corrected and fuel-corrected pace
- TrackAnalytics: Track evolution, grip progression, weather impact

Usage:
    from src.analysis import TyreAnalytics, FuelAnalytics, LapAnalytics

    # Tyre analysis
    tyre = TyreAnalytics(session_id=1)
    wear = tyre.calculate_wear_rate()

    # Lap analysis
    lap = LapAnalytics(session_id=1)
    ideal = lap.calculate_ideal_lap()
    consistency = lap.calculate_consistency()

    # Pace analysis
    pace = PaceAnalytics(session_id=1)
    race_pace = pace.calculate_race_pace()
"""

from .tyre_analytics import TyreAnalytics, TyreWearAnalysis, TyreTempAnalysis
from .fuel_analytics import FuelAnalytics, FuelConsumptionAnalysis, StintPlanningAnalysis
from .lap_analytics import LapAnalytics, IdealLapAnalysis, ConsistencyAnalysis
from .pace_analytics import PaceAnalytics, RacePaceAnalysis
from .track_analytics import TrackAnalytics, TrackEvolutionAnalysis, WeatherImpactAnalysis

__all__ = [
    'TyreAnalytics',
    'TyreWearAnalysis',
    'TyreTempAnalysis',
    'FuelAnalytics',
    'FuelConsumptionAnalysis',
    'StintPlanningAnalysis',
    'LapAnalytics',
    'IdealLapAnalysis',
    'ConsistencyAnalysis',
    'PaceAnalytics',
    'RacePaceAnalysis',
    'TrackAnalytics',
    'TrackEvolutionAnalysis',
    'WeatherImpactAnalysis'
]
