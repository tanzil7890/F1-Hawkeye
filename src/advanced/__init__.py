"""
F1 Telemetry Advanced Features Module

 advanced features for serious analysis.

Modules:
- AnomalyDetector: Detect unusual patterns in telemetry
- MultiSessionLearning: Track and driver-specific learning
- StrategyAlerts: Real-time strategy recommendations

Usage:
    from src.advanced import AnomalyDetector, MultiSessionLearning

    detector = AnomalyDetector()
    anomalies = detector.detect_tyre_anomalies(session_id=1)
"""

from .anomaly_detection import AnomalyDetector
from .multi_session_learning import MultiSessionLearning

__all__ = [
    'AnomalyDetector',
    'MultiSessionLearning'
]
