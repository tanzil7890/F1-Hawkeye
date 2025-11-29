"""
F1 Telemetry Database Layer
Provides persistent storage for telemetry data, enabling historical analysis and ML training.
"""

from .db_manager import DatabaseManager
from .models import (
    Base,
    SessionModel,
    LapModel,
    TelemetrySnapshotModel,
    TyreDataModel,
    DamageEventModel,
    PitStopModel,
    WeatherSampleModel
)

__all__ = [
    'DatabaseManager',
    'Base',
    'SessionModel',
    'LapModel',
    'TelemetrySnapshotModel',
    'TyreDataModel',
    'DamageEventModel',
    'PitStopModel',
    'WeatherSampleModel'
]

__version__ = '1.0.0'
