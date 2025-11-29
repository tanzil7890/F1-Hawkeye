"""
Pit Stop Optimizer

Determines optimal pit stop timing using Monte Carlo simulation.

Inputs: Current position, tyre wear, fuel, competitors' strategies
Outputs: Optimal pit lap, expected position after stop
Algorithm: Monte Carlo simulation with tyre degradation modeling
Value: "Pit lap 18 → P5 (vs lap 20 → P7)"

Usage:
    from src.ml.models import PitStopOptimizer

    optimizer = PitStopOptimizer()
    result = optimizer.optimize_pit_window(
        current_state={
            'current_lap': 10,
            'current_position': 5,
            'tyre_age': 10,
            'total_laps': 50,
            'competitors': {...}
        }
    )
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PitStopScenario:
    """Result of a pit stop scenario simulation"""
    pit_lap: int
    expected_position: float
    expected_race_time: float
    tyre_life_at_end: float
    undercut_potential: float
    overcut_potential: float
    risk_score: float


class PitStopOptimizer:
    """
    Pit stop timing optimizer

    Uses Monte Carlo simulation to find optimal pit window.
    """

    def __init__(
        self,
        pit_loss_time: float = 25000,  # ms (typical ~25s)
        degradation_per_lap: float = 50,  # ms per lap
        fresh_tyre_advantage: float = 1500,  # ms per lap on fresh tyres
        max_tyre_life: int = 40  # laps
    ):
        """Initialize pit stop optimizer"""
        self.pit_loss_time = pit_loss_time
        self.degradation_per_lap = degradation_per_lap
        self.fresh_tyre_advantage = fresh_tyre_advantage
        self.max_tyre_life = max_tyre_life

    def optimize_pit_window(
        self,
        current_state: Dict,
        candidate_laps: Optional[List[int]] = None,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Find optimal pit stop lap

        Args:
            current_state: Current race state
            candidate_laps: List of laps to consider (None = auto-generate)
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Optimization result with recommended pit lap
        """
        current_lap = current_state['current_lap']
        total_laps = current_state['total_laps']
        tyre_age = current_state.get('tyre_age', 0)
        current_position = current_state.get('current_position', 10)

        # Generate candidate pit laps if not provided
        if candidate_laps is None:
            min_lap = max(current_lap + 3, 8)  # At least 3 laps ahead, min lap 8
            max_lap = min(total_laps - 5, current_lap + 25)  # At least 5 laps left
            candidate_laps = list(range(min_lap, max_lap + 1, 2))  # Every 2 laps

        print(f"[PitStopOptimizer] Evaluating {len(candidate_laps)} pit windows...")

        # Simulate each candidate lap
        scenarios = []
        for pit_lap in candidate_laps:
            scenario = self._simulate_pit_stop(
                current_state,
                pit_lap,
                num_simulations
            )
            scenarios.append(scenario)

        # Find optimal scenario (minimize race time, then position)
        scenarios.sort(key=lambda s: (s.expected_position, s.expected_race_time))
        optimal = scenarios[0]

        # Calculate alternatives
        alternatives = []
        for i, scenario in enumerate(scenarios[:5]):  # Top 5 alternatives
            alternatives.append({
                'pit_lap': scenario.pit_lap,
                'expected_position': round(scenario.expected_position, 1),
                'expected_race_time_s': scenario.expected_race_time / 1000,
                'tyre_life_remaining': round(scenario.tyre_life_at_end, 1),
                'undercut_potential': round(scenario.undercut_potential, 2),
                'overcut_potential': round(scenario.overcut_potential, 2),
                'risk_score': round(scenario.risk_score, 2)
            })

        # Build result
        result = {
            'optimal_pit_lap': optimal.pit_lap,
            'expected_position': round(optimal.expected_position, 1),
            'current_position': current_position,
            'position_change': round(current_position - optimal.expected_position, 1),
            'expected_race_time_s': optimal.expected_race_time / 1000,
            'tyre_life_at_end': round(optimal.tyre_life_at_end, 1),
            'undercut_potential': round(optimal.undercut_potential, 2),
            'overcut_potential': round(optimal.overcut_potential, 2),
            'risk_score': round(optimal.risk_score, 2),
            'recommendation': self._generate_recommendation(optimal, current_position),
            'alternatives': alternatives
        }

        print(f"[PitStopOptimizer] Optimal: Lap {optimal.pit_lap} → P{optimal.expected_position:.1f}")

        return result

    def _simulate_pit_stop(
        self,
        current_state: Dict,
        pit_lap: int,
        num_simulations: int
    ) -> PitStopScenario:
        """
        Simulate a pit stop at specific lap

        Uses Monte Carlo to account for variability.
        """
        current_lap = current_state['current_lap']
        total_laps = current_state['total_laps']
        tyre_age = current_state.get('tyre_age', 0)
        current_position = current_state.get('current_position', 10)
        base_lap_time = current_state.get('base_lap_time', 90000)  # 1:30 default

        # Laps until pit
        laps_to_pit = pit_lap - current_lap

        # Laps after pit
        laps_after_pit = total_laps - pit_lap

        # Run simulations
        positions = []
        race_times = []

        for _ in range(num_simulations):
            # Phase 1: Before pit (on current tyres)
            tyre_age_at_pit = tyre_age + laps_to_pit
            phase1_time = self._calculate_stint_time(
                laps_to_pit,
                tyre_age,
                base_lap_time
            )

            # Pit stop
            pit_time = self.pit_loss_time + np.random.normal(0, 500)  # +/- 0.5s variability

            # Phase 2: After pit (on fresh tyres)
            phase2_time = self._calculate_stint_time(
                laps_after_pit,
                0,  # Fresh tyres
                base_lap_time - self.fresh_tyre_advantage
            )

            # Total race time
            total_time = phase1_time + pit_time + phase2_time
            race_times.append(total_time)

            # Estimate position (simplified model)
            # Position loss from pit stop
            position_loss = self.pit_loss_time / base_lap_time

            # Position gain from fresh tyres
            position_gain = (self.fresh_tyre_advantage * laps_after_pit) / base_lap_time

            # Net position change
            net_change = position_gain - position_loss

            # Add variability
            net_change += np.random.normal(0, 0.5)

            estimated_position = max(1, current_position - net_change)
            positions.append(estimated_position)

        # Calculate scenario metrics
        expected_position = np.mean(positions)
        expected_race_time = np.mean(race_times)

        # Tyre life at end
        tyre_life_at_end = max(0, self.max_tyre_life - laps_after_pit)

        # Undercut potential (earlier pit = higher undercut)
        undercut_potential = max(0, 1 - (laps_to_pit / 15))

        # Overcut potential (later pit = higher overcut)
        overcut_potential = max(0, min(1, laps_to_pit / 20))

        # Risk score (tyre not lasting, traffic, etc.)
        risk_score = self._calculate_risk(
            laps_after_pit,
            tyre_life_at_end,
            expected_position
        )

        return PitStopScenario(
            pit_lap=pit_lap,
            expected_position=expected_position,
            expected_race_time=expected_race_time,
            tyre_life_at_end=tyre_life_at_end,
            undercut_potential=undercut_potential,
            overcut_potential=overcut_potential,
            risk_score=risk_score
        )

    def _calculate_stint_time(
        self,
        num_laps: int,
        starting_tyre_age: int,
        base_lap_time: float
    ) -> float:
        """Calculate total time for a stint with tyre degradation"""
        total_time = 0

        for lap in range(num_laps):
            current_tyre_age = starting_tyre_age + lap

            # Degradation effect (quadratic after certain age)
            if current_tyre_age < 10:
                degradation = self.degradation_per_lap * current_tyre_age
            else:
                # Accelerated degradation
                degradation = self.degradation_per_lap * (
                    10 + 1.5 * (current_tyre_age - 10)
                )

            lap_time = base_lap_time + degradation
            total_time += lap_time

        return total_time

    def _calculate_risk(
        self,
        laps_after_pit: int,
        tyre_life_at_end: float,
        expected_position: float
    ) -> float:
        """
        Calculate risk score for pit strategy

        Higher risk = more uncertainty/danger
        """
        risk = 0.0

        # Risk from running out of tyre life
        if tyre_life_at_end < 5:
            risk += 0.5
        elif tyre_life_at_end < 10:
            risk += 0.2

        # Risk from very long second stint
        if laps_after_pit > 30:
            risk += 0.3
        elif laps_after_pit > 25:
            risk += 0.1

        # Risk from being in traffic (positions 6-15)
        if 6 <= expected_position <= 15:
            risk += 0.2

        return min(1.0, risk)

    def _generate_recommendation(
        self,
        scenario: PitStopScenario,
        current_position: int
    ) -> str:
        """Generate human-readable recommendation"""
        position_change = current_position - scenario.expected_position

        if position_change > 1:
            position_text = f"gain {position_change:.0f} positions"
        elif position_change < -1:
            position_text = f"lose {abs(position_change):.0f} positions"
        else:
            position_text = "maintain position"

        if scenario.undercut_potential > 0.7:
            strategy_type = "aggressive undercut"
        elif scenario.overcut_potential > 0.7:
            strategy_type = "overcut strategy"
        else:
            strategy_type = "standard stop"

        return (
            f"Pit on lap {scenario.pit_lap} ({strategy_type}). "
            f"Expected to {position_text} → P{scenario.expected_position:.0f}. "
            f"Risk level: {'HIGH' if scenario.risk_score > 0.5 else 'LOW'}."
        )

    def evaluate_competitor_strategies(
        self,
        current_state: Dict,
        competitors: List[Dict]
    ) -> Dict:
        """
        Evaluate pit strategies relative to competitors

        Args:
            current_state: Your current state
            competitors: List of competitor states

        Returns:
            Strategic analysis relative to field
        """
        # Get your optimal strategy
        your_strategy = self.optimize_pit_window(current_state)

        # Simulate competitors
        competitor_strategies = []
        for comp in competitors:
            comp_result = self.optimize_pit_window(comp, num_simulations=500)
            competitor_strategies.append({
                'driver': comp.get('driver_name', 'Unknown'),
                'position': comp.get('current_position', 0),
                'optimal_pit_lap': comp_result['optimal_pit_lap'],
                'expected_position': comp_result['expected_position']
            })

        # Identify undercut/overcut opportunities
        opportunities = []
        your_pit_lap = your_strategy['optimal_pit_lap']

        for comp in competitor_strategies:
            if abs(comp['optimal_pit_lap'] - your_pit_lap) <= 3:
                if comp['optimal_pit_lap'] > your_pit_lap:
                    opportunities.append({
                        'driver': comp['driver'],
                        'opportunity': 'undercut',
                        'pit_lap_delta': comp['optimal_pit_lap'] - your_pit_lap
                    })
                elif comp['optimal_pit_lap'] < your_pit_lap:
                    opportunities.append({
                        'driver': comp['driver'],
                        'opportunity': 'overcut',
                        'pit_lap_delta': your_pit_lap - comp['optimal_pit_lap']
                    })

        return {
            'your_strategy': your_strategy,
            'competitor_strategies': competitor_strategies,
            'opportunities': opportunities,
            'strategic_summary': self._generate_strategic_summary(
                your_strategy,
                opportunities
            )
        }

    def _generate_strategic_summary(
        self,
        your_strategy: Dict,
        opportunities: List[Dict]
    ) -> str:
        """Generate strategic summary"""
        if not opportunities:
            return f"Box lap {your_strategy['optimal_pit_lap']}. No direct battles expected."

        undercuts = [o for o in opportunities if o['opportunity'] == 'undercut']
        overcuts = [o for o in opportunities if o['opportunity'] == 'overcut']

        summary = f"Box lap {your_strategy['optimal_pit_lap']}. "

        if undercuts:
            summary += f"Undercut opportunities on {len(undercuts)} car(s). "
        if overcuts:
            summary += f"At risk of overcut from {len(overcuts)} car(s). "

        return summary
