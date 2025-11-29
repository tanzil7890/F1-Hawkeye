"""
F1 Telemetry Live Strategy Alerts Module

Real-time strategic alerts and recommendations.

Modules:
- StrategyAlertsEngine: Live strategy recommendations

Usage:
    from src.alerts import StrategyAlertsEngine

    alerts = StrategyAlertsEngine()
    current_alerts = alerts.check_all_alerts(current_state)
"""

from .strategy_alerts import StrategyAlertsEngine

__all__ = ['StrategyAlertsEngine']
