"""
Strategy Recommendation Model

Combines all ML models to provide comprehensive race strategy recommendations.

Combines:
- Tyre wear model (degradation predictions)
- Lap time model (pace forecasting)
- Pit stop optimizer (optimal timing)
- Race outcome model (win/podium probabilities)

Output: "1-stop Medium-Hard: P4 (70%), 2-stop Soft-Medium-Soft: P5 (60%)"
Value: Strategic decision support in real-time

Usage:
    from src.ml.models import StrategyRecommender

    recommender = StrategyRecommender()
    strategy = recommender.recommend_strategy(current_state)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .tyre_wear_model import TyreWearModel
from .lap_time_model import LapTimeModel
from .pit_stop_optimizer import PitStopOptimizer
from .race_outcome_model import RaceOutcomeModel


@dataclass
class StrategyOption:
    """A race strategy option"""
    name: str
    pit_stops: int
    compounds: List[str]
    pit_laps: List[int]
    expected_position: int
    win_probability: float
    podium_probability: float
    points_probability: float
    total_race_time: float
    risk_level: str
    confidence: float
    reasoning: str


class StrategyRecommender:
    """
    Strategy recommendation engine

    Combines multiple ML models for comprehensive race strategy.
    """

    def __init__(
        self,
        tyre_model: Optional[TyreWearModel] = None,
        lap_time_model: Optional[LapTimeModel] = None,
        pit_optimizer: Optional[PitStopOptimizer] = None,
        outcome_model: Optional[RaceOutcomeModel] = None
    ):
        """Initialize strategy recommender"""
        self.tyre_model = tyre_model or TyreWearModel()
        self.lap_time_model = lap_time_model or LapTimeModel()
        self.pit_optimizer = pit_optimizer or PitStopOptimizer()
        self.outcome_model = outcome_model or RaceOutcomeModel()

        # Tyre compound characteristics (approximate)
        self.compound_characteristics = {
            'SOFT': {'initial_pace': -0.8, 'degradation_rate': 100, 'max_stint': 20},
            'MEDIUM': {'initial_pace': -0.3, 'degradation_rate': 60, 'max_stint': 35},
            'HARD': {'initial_pace': 0.0, 'degradation_rate': 40, 'max_stint': 50}
        }

    def recommend_strategy(
        self,
        current_state: Dict,
        max_strategies: int = 5
    ) -> Dict:
        """
        Recommend race strategies

        Args:
            current_state: Current race state
            max_strategies: Maximum number of strategies to return

        Returns:
            Strategy recommendations with analysis
        """
        print(f"[StrategyRecommender] Analyzing race strategies...")

        current_lap = current_state.get('current_lap', 0)
        total_laps = current_state.get('total_laps', 50)
        current_position = current_state.get('current_position', 10)
        current_compound = current_state.get('current_compound', 'MEDIUM')

        # Generate candidate strategies
        strategies = []

        # 1-stop strategies
        strategies.extend(self._generate_one_stop_strategies(current_state))

        # 2-stop strategies (if race is long enough)
        if total_laps >= 40:
            strategies.extend(self._generate_two_stop_strategies(current_state))

        # 0-stop strategy (if possible)
        if self._is_zero_stop_viable(current_state):
            strategies.append(self._generate_zero_stop_strategy(current_state))

        # Evaluate each strategy
        evaluated_strategies = []
        for strategy in strategies:
            evaluation = self._evaluate_strategy(strategy, current_state)
            evaluated_strategies.append(evaluation)

        # Sort by expected outcome (podium prob, then position)
        evaluated_strategies.sort(
            key=lambda s: (s.podium_probability, -s.expected_position),
            reverse=True
        )

        # Take top strategies
        top_strategies = evaluated_strategies[:max_strategies]

        # Generate recommendation
        recommendation = self._generate_recommendation(
            top_strategies,
            current_state
        )

        return {
            'recommended_strategy': top_strategies[0] if top_strategies else None,
            'alternative_strategies': top_strategies[1:] if len(top_strategies) > 1 else [],
            'recommendation_text': recommendation,
            'current_state': current_state,
            'analysis': {
                'total_strategies_evaluated': len(strategies),
                'best_pit_window': top_strategies[0].pit_laps[0] if top_strategies and top_strategies[0].pit_laps else None,
                'risk_assessment': self._assess_overall_risk(current_state)
            }
        }

    def _generate_one_stop_strategies(self, state: Dict) -> List[Dict]:
        """Generate 1-stop strategy candidates"""
        strategies = []

        current_lap = state.get('current_lap', 0)
        total_laps = state.get('total_laps', 50)
        current_compound = state.get('current_compound', 'MEDIUM')

        # Available compounds (must use 2 different compounds in race)
        available_compounds = [c for c in ['SOFT', 'MEDIUM', 'HARD'] if c != current_compound]

        # Pit window options
        min_pit_lap = max(current_lap + 3, 10)
        max_pit_lap = min(total_laps - 10, current_lap + 30)

        for target_compound in available_compounds:
            # Early stop (laps 12-18)
            if min_pit_lap <= 18:
                strategies.append({
                    'name': f'1-stop {current_compound}-{target_compound} (Early)',
                    'pit_stops': 1,
                    'compounds': [current_compound, target_compound],
                    'pit_laps': [min(18, max_pit_lap)]
                })

            # Mid stop (laps 22-28)
            if min_pit_lap <= 28 and max_pit_lap >= 22:
                strategies.append({
                    'name': f'1-stop {current_compound}-{target_compound} (Mid)',
                    'pit_stops': 1,
                    'compounds': [current_compound, target_compound],
                    'pit_laps': [min(25, max_pit_lap)]
                })

            # Late stop (laps 30-36)
            if min_pit_lap <= 36 and max_pit_lap >= 30:
                strategies.append({
                    'name': f'1-stop {current_compound}-{target_compound} (Late)',
                    'pit_stops': 1,
                    'compounds': [current_compound, target_compound],
                    'pit_laps': [min(33, max_pit_lap)]
                })

        return strategies

    def _generate_two_stop_strategies(self, state: Dict) -> List[Dict]:
        """Generate 2-stop strategy candidates"""
        strategies = []

        current_lap = state.get('current_lap', 0)
        total_laps = state.get('total_laps', 50)
        current_compound = state.get('current_compound', 'MEDIUM')

        # Only consider 2-stop if enough laps remaining
        laps_remaining = total_laps - current_lap
        if laps_remaining < 30:
            return strategies

        # Common 2-stop strategies
        two_stop_patterns = [
            {
                'name': '2-stop Soft-Medium-Soft (Aggressive)',
                'compounds': ['SOFT', 'MEDIUM', 'SOFT'],
                'pit_laps': [15, 35]
            },
            {
                'name': '2-stop Medium-Medium-Soft (Balanced)',
                'compounds': ['MEDIUM', 'MEDIUM', 'SOFT'],
                'pit_laps': [18, 36]
            },
            {
                'name': '2-stop Medium-Soft-Medium (Flexible)',
                'compounds': ['MEDIUM', 'SOFT', 'MEDIUM'],
                'pit_laps': [16, 32]
            }
        ]

        for pattern in two_stop_patterns:
            # Adjust pit laps if already past suggested laps
            adjusted_pit_laps = []
            for lap in pattern['pit_laps']:
                if lap > current_lap + 2:
                    adjusted_pit_laps.append(lap)
                else:
                    adjusted_pit_laps.append(current_lap + 5)

            if len(adjusted_pit_laps) == 2:
                strategies.append({
                    'name': pattern['name'],
                    'pit_stops': 2,
                    'compounds': pattern['compounds'],
                    'pit_laps': adjusted_pit_laps
                })

        return strategies

    def _generate_zero_stop_strategy(self, state: Dict) -> Dict:
        """Generate 0-stop strategy (if viable)"""
        current_compound = state.get('current_compound', 'MEDIUM')

        return {
            'name': f'0-stop {current_compound} (No Stop)',
            'pit_stops': 0,
            'compounds': [current_compound],
            'pit_laps': []
        }

    def _is_zero_stop_viable(self, state: Dict) -> bool:
        """Check if 0-stop strategy is viable"""
        tyre_age = state.get('tyre_age', 0)
        total_laps = state.get('total_laps', 50)
        current_lap = state.get('current_lap', 0)
        current_compound = state.get('current_compound', 'MEDIUM')

        laps_remaining = total_laps - current_lap
        projected_tyre_age = tyre_age + laps_remaining

        # Check compound max stint
        max_stint = self.compound_characteristics.get(current_compound, {}).get('max_stint', 30)

        return projected_tyre_age <= max_stint

    def _evaluate_strategy(
        self,
        strategy: Dict,
        current_state: Dict
    ) -> StrategyOption:
        """Evaluate a strategy option"""
        # Simulate the strategy
        race_time = self._simulate_strategy_race_time(strategy, current_state)
        expected_position = self._simulate_strategy_position(strategy, current_state)

        # Predict outcome probabilities
        # Create state for outcome prediction
        outcome_state = {
            **current_state,
            'expected_position': expected_position,
            'strategy_risk': self._calculate_strategy_risk(strategy, current_state)
        }

        # Calculate outcome probabilities (simplified)
        if expected_position <= 3:
            podium_prob = 0.75
            win_prob = 0.25 if expected_position == 1 else 0.05
        elif expected_position <= 6:
            podium_prob = 0.40
            win_prob = 0.02
        elif expected_position <= 10:
            podium_prob = 0.15
            win_prob = 0.01
        else:
            podium_prob = 0.05
            win_prob = 0.001

        points_prob = 0.95 if expected_position <= 10 else 0.30

        # Risk level
        risk_level = self._calculate_risk_level(strategy, current_state)

        # Confidence
        confidence = self._calculate_strategy_confidence(strategy, current_state)

        # Reasoning
        reasoning = self._generate_strategy_reasoning(strategy, current_state, expected_position)

        return StrategyOption(
            name=strategy['name'],
            pit_stops=strategy['pit_stops'],
            compounds=strategy['compounds'],
            pit_laps=strategy['pit_laps'],
            expected_position=expected_position,
            win_probability=win_prob,
            podium_probability=podium_prob,
            points_probability=points_prob,
            total_race_time=race_time,
            risk_level=risk_level,
            confidence=confidence,
            reasoning=reasoning
        )

    def _simulate_strategy_race_time(self, strategy: Dict, state: Dict) -> float:
        """Simulate total race time for a strategy"""
        total_laps = state.get('total_laps', 50)
        current_lap = state.get('current_lap', 0)
        base_lap_time = state.get('base_lap_time', 90000)  # 1:30 default

        laps_remaining = total_laps - current_lap
        total_time = 0

        # Simulate each stint
        pit_laps = strategy['pit_laps']
        compounds = strategy['compounds']

        stint_start_lap = current_lap
        compound_index = 0

        for i, pit_lap in enumerate(pit_laps + [total_laps]):
            stint_length = pit_lap - stint_start_lap
            compound = compounds[compound_index]

            # Stint time
            compound_char = self.compound_characteristics.get(compound, {})
            pace_delta = compound_char.get('initial_pace', 0)
            degradation = compound_char.get('degradation_rate', 60)

            stint_time = self._calculate_stint_time(
                stint_length,
                base_lap_time + (pace_delta * 1000),
                degradation
            )

            total_time += stint_time

            # Pit stop time (if not last stint)
            if pit_lap < total_laps:
                total_time += 25000  # 25s pit stop

            stint_start_lap = pit_lap
            compound_index += 1

        return total_time

    def _calculate_stint_time(
        self,
        laps: int,
        base_lap_time: float,
        degradation_per_lap: float
    ) -> float:
        """Calculate stint time with degradation"""
        total = 0
        for lap in range(laps):
            lap_time = base_lap_time + (degradation_per_lap * lap)
            total += lap_time
        return total

    def _simulate_strategy_position(self, strategy: Dict, state: Dict) -> int:
        """Simulate expected position for strategy"""
        current_position = state.get('current_position', 10)

        # Simplified position change based on pit stops
        # More pit stops = higher risk but potentially better pace
        if strategy['pit_stops'] == 0:
            # 0-stop: lose positions due to tyre degradation
            position_change = 2
        elif strategy['pit_stops'] == 1:
            # 1-stop: standard
            position_change = 0
        elif strategy['pit_stops'] == 2:
            # 2-stop: gain positions with fresher tyres
            position_change = -1
        else:
            position_change = 1

        # Adjust for compound aggressiveness
        if 'SOFT' in strategy['compounds']:
            position_change -= 0.5  # Softs = faster

        expected_position = current_position + position_change

        return max(1, min(20, int(round(expected_position))))

    def _calculate_strategy_risk(self, strategy: Dict, state: Dict) -> float:
        """Calculate strategy risk score"""
        risk = 0.0

        # More pit stops = more risk
        risk += strategy['pit_stops'] * 0.2

        # Soft tyres = more risk (degradation)
        if 'SOFT' in strategy['compounds']:
            risk += 0.3

        # Long stints = more risk
        if strategy['pit_laps']:
            for i, pit_lap in enumerate(strategy['pit_laps'] + [state.get('total_laps', 50)]):
                start_lap = strategy['pit_laps'][i - 1] if i > 0 else state.get('current_lap', 0)
                stint_length = pit_lap - start_lap
                if stint_length > 30:
                    risk += 0.2

        return min(1.0, risk)

    def _calculate_risk_level(self, strategy: Dict, state: Dict) -> str:
        """Calculate risk level string"""
        risk = self._calculate_strategy_risk(strategy, state)

        if risk < 0.3:
            return 'LOW'
        elif risk < 0.6:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _calculate_strategy_confidence(self, strategy: Dict, state: Dict) -> float:
        """Calculate confidence in strategy recommendation"""
        # Higher confidence for standard strategies
        if strategy['pit_stops'] == 1:
            return 0.8
        elif strategy['pit_stops'] == 0 or strategy['pit_stops'] == 2:
            return 0.6
        else:
            return 0.4

    def _generate_strategy_reasoning(
        self,
        strategy: Dict,
        state: Dict,
        expected_position: int
    ) -> str:
        """Generate reasoning for strategy"""
        parts = []

        # Position expectation
        current_pos = state.get('current_position', 10)
        if expected_position < current_pos:
            parts.append(f"Expected to gain {current_pos - expected_position} position(s)")
        elif expected_position > current_pos:
            parts.append(f"May lose {expected_position - current_pos} position(s)")
        else:
            parts.append("Maintain current position")

        # Pit stop timing
        if strategy['pit_laps']:
            parts.append(f"Pit lap(s): {', '.join(map(str, strategy['pit_laps']))}")

        # Compound strategy
        compounds_str = '-'.join(strategy['compounds'])
        parts.append(f"Tyres: {compounds_str}")

        return '. '.join(parts) + '.'

    def _assess_overall_risk(self, state: Dict) -> str:
        """Assess overall race risk"""
        position = state.get('current_position', 10)
        lap = state.get('current_lap', 0)
        total = state.get('total_laps', 50)

        if lap < total * 0.3:
            return 'Early race - conservative approach recommended'
        elif lap < total * 0.7:
            return 'Mid race - strategic window open'
        else:
            return 'Late race - limited strategic options'

    def _generate_recommendation(
        self,
        strategies: List[StrategyOption],
        state: Dict
    ) -> str:
        """Generate human-readable recommendation"""
        if not strategies:
            return "No viable strategies found."

        best = strategies[0]

        rec = f"Recommended: {best.name}\n"
        rec += f"Expected outcome: P{best.expected_position}\n"
        rec += f"Podium probability: {best.podium_probability * 100:.0f}%\n"
        rec += f"Risk level: {best.risk_level}\n"
        rec += f"Reasoning: {best.reasoning}"

        if len(strategies) > 1:
            rec += f"\n\nAlternative: {strategies[1].name} (P{strategies[1].expected_position})"

        return rec
