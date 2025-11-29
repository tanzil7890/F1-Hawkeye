"""
Live Strategy Alerts Engine

Real-time alerts for strategic decisions during the race.

Alert Types:
- Pit window alerts (undercut/overcut opportunities)
- Weather change alerts
- Competitor strategy alerts
- Tyre management warnings
- DRS train formation alerts

Usage:
    from src.alerts import StrategyAlertsEngine

    engine = StrategyAlertsEngine()
    alerts = engine.check_all_alerts({
        'current_lap': 25,
        'current_position': 5,
        'tyre_age': 20,
        'avg_wear': 65.0
    })

    for alert in alerts:
        print(f"{alert['severity']}: {alert['message']}")
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class StrategyAlert:
    """Strategic alert"""
    severity: AlertSeverity
    type: str
    message: str
    recommendation: str
    lap: int


class StrategyAlertsEngine:
    """
    Live strategy alerts engine

    Monitors race state and generates real-time alerts.
    """

    def __init__(self):
        """Initialize alerts engine"""
        self.alert_history = []

    def check_all_alerts(self, current_state: Dict) -> List[Dict]:
        """
        Check all alert conditions

        Args:
            current_state: Current race state

        Returns:
            List of active alerts
        """
        alerts = []

        # Pit window alerts
        alerts.extend(self._check_pit_window(current_state))

        # Tyre management alerts
        alerts.extend(self._check_tyre_condition(current_state))

        # Position alerts
        alerts.extend(self._check_position_alerts(current_state))

        # Store in history
        self.alert_history.extend(alerts)

        return alerts

    def _check_pit_window(self, state: Dict) -> List[Dict]:
        """Check for pit window opportunities"""
        alerts = []

        tyre_age = state.get('tyre_age', 0)
        avg_wear = state.get('avg_wear', 0)
        current_lap = state.get('current_lap', 0)

        # Undercut window
        if 15 <= tyre_age <= 20 and avg_wear < 60:
            alerts.append({
                'severity': 'WARNING',
                'type': 'undercut_window',
                'message': f'⚠️  UNDERCUT WINDOW OPEN',
                'recommendation': f'Pit now to undercut competitors (tyre age: {tyre_age} laps, {avg_wear:.0f}% wear)',
                'lap': current_lap
            })

        # Overcut opportunity
        if tyre_age > 25 and avg_wear < 75:
            alerts.append({
                'severity': 'INFO',
                'type': 'overcut_opportunity',
                'message': f'ℹ️  OVERCUT OPPORTUNITY',
                'recommendation': f'Stay out longer for overcut strategy',
                'lap': current_lap
            })

        return alerts

    def _check_tyre_condition(self, state: Dict) -> List[Dict]:
        """Check tyre condition warnings"""
        alerts = []

        avg_wear = state.get('avg_wear', 0)
        tyre_age = state.get('tyre_age', 0)
        current_lap = state.get('current_lap', 0)

        # Critical wear
        if avg_wear > 85:
            alerts.append({
                'severity': 'CRITICAL',
                'type': 'critical_tyre_wear',
                'message': f'🚨 CRITICAL TYRE WEAR',
                'recommendation': f'PIT IMMEDIATELY - {avg_wear:.0f}% wear!',
                'lap': current_lap
            })

        # High wear warning
        elif avg_wear > 75:
            alerts.append({
                'severity': 'WARNING',
                'type': 'high_tyre_wear',
                'message': f'⚠️  HIGH TYRE WEAR',
                'recommendation': f'Plan pit stop within 3-5 laps ({avg_wear:.0f}% wear)',
                'lap': current_lap
            })

        # Long stint warning
        if tyre_age > 35:
            alerts.append({
                'severity': 'WARNING',
                'type': 'long_stint',
                'message': f'⚠️  LONG STINT',
                'recommendation': f'Tyre age {tyre_age} laps - consider pitting soon',
                'lap': current_lap
            })

        return alerts

    def _check_position_alerts(self, state: Dict) -> List[Dict]:
        """Check position-related alerts"""
        alerts = []

        current_position = state.get('current_position', 10)
        current_lap = state.get('current_lap', 0)

        # Podium position
        if current_position <= 3:
            alerts.append({
                'severity': 'INFO',
                'type': 'podium_position',
                'message': f'🏆 PODIUM POSITION',
                'recommendation': f'In P{current_position} - protect position',
                'lap': current_lap
            })

        return alerts

    def check_competitor_action(
        self,
        competitor_action: str,
        current_state: Dict
    ) -> List[Dict]:
        """
        React to competitor actions

        Args:
            competitor_action: Action type ('pit', 'crash', 'slow')
            current_state: Current race state

        Returns:
            List of recommended responses
        """
        alerts = []
        current_lap = current_state.get('current_lap', 0)

        if competitor_action == 'pit':
            # Competitor ahead pitting
            alerts.append({
                'severity': 'WARNING',
                'type': 'competitor_pit',
                'message': f'⚠️  COMPETITOR PITTING',
                'recommendation': 'Consider COVER STOP or STAY OUT for overcut',
                'lap': current_lap
            })

        elif competitor_action == 'crash':
            # Safety car likely
            alerts.append({
                'severity': 'CRITICAL',
                'type': 'safety_car_likely',
                'message': f'🚨 INCIDENT - SAFETY CAR LIKELY',
                'recommendation': 'BOX BOX BOX - Free pit stop opportunity!',
                'lap': current_lap
            })

        return alerts

    def format_alert_display(self, alerts: List[Dict]) -> str:
        """Format alerts for display"""
        if not alerts:
            return "✓ No active alerts"

        lines = []
        lines.append("=" * 60)
        lines.append("LIVE STRATEGY ALERTS")
        lines.append("=" * 60)

        for alert in alerts:
            lines.append(f"\n{alert['severity']}: {alert['message']}")
            lines.append(f"→ {alert['recommendation']}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
