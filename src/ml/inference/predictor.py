"""
Real-Time Prediction Engine

Orchestrates all ML models for live race predictions.

Features:
- Load all trained models
- Real-time predictions from current state
- Comprehensive strategy recommendations
- Performance monitoring

Usage:
    from src.ml.inference import RealTimePredictor

    # Initialize and load models
    predictor = RealTimePredictor()
    predictor.load_models('models/')

    # Get real-time predictions
    predictions = predictor.predict({
        'current_lap': 25,
        'current_position': 5,
        'tyre_age': 15,
        'total_laps': 50,
        ...
    })

    # Get strategy recommendation
    strategy = predictor.recommend_strategy(current_state)
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

from ..models import (
    TyreWearModel,
    LapTimeModel,
    PitStopOptimizer,
    RaceOutcomeModel,
    StrategyRecommender
)


class RealTimePredictor:
    """
    Real-time prediction engine

    Orchestrates all ML models for live predictions.
    """

    def __init__(self):
        """Initialize predictor"""
        self.tyre_model: Optional[TyreWearModel] = None
        self.lap_time_model: Optional[LapTimeModel] = None
        self.pit_optimizer = PitStopOptimizer()  # No training needed
        self.outcome_model: Optional[RaceOutcomeModel] = None
        self.strategy_recommender: Optional[StrategyRecommender] = None

        self.models_loaded = False
        self.prediction_history = []

    def load_models(self, models_dir: str):
        """
        Load trained models from directory

        Args:
            models_dir: Directory containing model files
        """
        print(f"[RealTimePredictor] Loading models from {models_dir}...")

        # Load tyre wear model
        tyre_path = os.path.join(models_dir, 'tyre_wear_model.pkl')
        if os.path.exists(tyre_path):
            try:
                self.tyre_model = TyreWearModel(model_path=tyre_path)
                print(f"  ✓ Loaded tyre wear model")
            except Exception as e:
                print(f"  ✗ Failed to load tyre wear model: {e}")

        # Load lap time model
        lap_path = os.path.join(models_dir, 'lap_time_model.pkl')
        if os.path.exists(lap_path):
            try:
                self.lap_time_model = LapTimeModel(model_path=lap_path)
                print(f"  ✓ Loaded lap time model")
            except Exception as e:
                print(f"  ✗ Failed to load lap time model: {e}")

        # Load race outcome model
        outcome_path = os.path.join(models_dir, 'race_outcome_model.pkl')
        if os.path.exists(outcome_path):
            try:
                self.outcome_model = RaceOutcomeModel(model_path=outcome_path)
                print(f"  ✓ Loaded race outcome model")
            except Exception as e:
                print(f"  ✗ Failed to load race outcome model: {e}")

        # Initialize strategy recommender with loaded models
        self.strategy_recommender = StrategyRecommender(
            tyre_model=self.tyre_model,
            lap_time_model=self.lap_time_model,
            pit_optimizer=self.pit_optimizer,
            outcome_model=self.outcome_model
        )

        self.models_loaded = True
        print(f"[RealTimePredictor] Models loaded successfully")

    def predict(self, current_state: Dict) -> Dict:
        """
        Make real-time predictions

        Args:
            current_state: Current race state

        Returns:
            Comprehensive predictions from all models
        """
        timestamp = datetime.now()

        predictions = {
            'timestamp': timestamp.isoformat(),
            'current_state': current_state,
            'tyre_wear': None,
            'lap_time': None,
            'pit_stop': None,
            'race_outcome': None,
            'errors': []
        }

        # Tyre wear prediction
        if self.tyre_model:
            try:
                tyre_forecast = self.tyre_model.predict_wear(
                    current_state,
                    future_laps=10
                )
                predictions['tyre_wear'] = tyre_forecast
            except Exception as e:
                predictions['errors'].append(f"Tyre wear prediction failed: {e}")

        # Lap time forecast
        if self.lap_time_model:
            try:
                lap_forecast = self.lap_time_model.forecast_lap_times(
                    current_state,
                    num_laps=5
                )
                predictions['lap_time'] = lap_forecast
            except Exception as e:
                predictions['errors'].append(f"Lap time forecast failed: {e}")

        # Pit stop optimization
        try:
            pit_strategy = self.pit_optimizer.optimize_pit_window(current_state)
            predictions['pit_stop'] = pit_strategy
        except Exception as e:
            predictions['errors'].append(f"Pit stop optimization failed: {e}")

        # Race outcome prediction
        if self.outcome_model:
            try:
                outcome = self.outcome_model.predict_outcome(current_state)
                predictions['race_outcome'] = {
                    'win_probability': outcome.win_probability,
                    'podium_probability': outcome.podium_probability,
                    'points_probability': outcome.points_probability,
                    'expected_position': outcome.expected_position,
                    'confidence': outcome.confidence,
                    'factors': outcome.factors
                }
            except Exception as e:
                predictions['errors'].append(f"Race outcome prediction failed: {e}")

        # Store in history
        self.prediction_history.append({
            'timestamp': timestamp,
            'lap': current_state.get('current_lap', 0),
            'predictions': predictions
        })

        return predictions

    def recommend_strategy(self, current_state: Dict, max_strategies: int = 5) -> Dict:
        """
        Get comprehensive strategy recommendation

        Args:
            current_state: Current race state
            max_strategies: Maximum number of strategies to return

        Returns:
            Strategy recommendations
        """
        if not self.strategy_recommender:
            return {'error': 'Strategy recommender not initialized'}

        try:
            strategy = self.strategy_recommender.recommend_strategy(
                current_state,
                max_strategies=max_strategies
            )
            return strategy
        except Exception as e:
            return {'error': f'Strategy recommendation failed: {e}'}

    def get_dashboard_summary(self, current_state: Dict) -> Dict:
        """
        Get dashboard-ready summary of all predictions

        Args:
            current_state: Current race state

        Returns:
            Dashboard summary
        """
        # Get predictions
        predictions = self.predict(current_state)

        # Build summary
        summary = {
            'current_lap': current_state.get('current_lap', 0),
            'current_position': current_state.get('current_position', 0),
            'tyre_age': current_state.get('tyre_age', 0),
            'predictions': {}
        }

        # Tyre wear summary
        if predictions['tyre_wear']:
            tw = predictions['tyre_wear']
            summary['predictions']['tyre_wear'] = {
                'next_lap_wear': f"{tw.get('next_lap_wear', 0):.1f}%",
                'laps_to_80_percent': tw.get('laps_to_80_percent', 0),
                'status': 'critical' if tw.get('laps_to_80_percent', 99) < 5 else 'good'
            }

        # Lap time summary
        if predictions['lap_time']:
            lt = predictions['lap_time']
            summary['predictions']['lap_time'] = {
                'next_lap': lt.get('next_lap_time_str', 'N/A'),
                'degradation_per_lap': f"{lt.get('degradation_per_lap_ms', 0) / 1000:.3f}s"
            }

        # Pit stop summary
        if predictions['pit_stop']:
            ps = predictions['pit_stop']
            summary['predictions']['pit_stop'] = {
                'optimal_lap': ps.get('optimal_pit_lap', 0),
                'expected_position': f"P{ps.get('expected_position', 0)}",
                'position_change': ps.get('position_change', 0)
            }

        # Race outcome summary
        if predictions['race_outcome']:
            ro = predictions['race_outcome']
            summary['predictions']['race_outcome'] = {
                'podium_probability': f"{ro.get('podium_probability', 0) * 100:.0f}%",
                'expected_position': f"P{ro.get('expected_position', 0)}",
                'confidence': f"{ro.get('confidence', 0) * 100:.0f}%"
            }

        return summary

    def get_prediction_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get prediction history

        Args:
            last_n: Return last N predictions (None = all)

        Returns:
            List of historical predictions
        """
        if last_n:
            return self.prediction_history[-last_n:]
        return self.prediction_history

    def clear_history(self):
        """Clear prediction history"""
        self.prediction_history = []
        print("[RealTimePredictor] Prediction history cleared")

    def get_status(self) -> Dict:
        """
        Get predictor status

        Returns:
            Status information
        """
        return {
            'models_loaded': self.models_loaded,
            'tyre_model_loaded': self.tyre_model is not None,
            'lap_time_model_loaded': self.lap_time_model is not None,
            'pit_optimizer_available': True,
            'outcome_model_loaded': self.outcome_model is not None,
            'strategy_recommender_available': self.strategy_recommender is not None,
            'predictions_in_history': len(self.prediction_history)
        }

    def live_monitoring_update(self, current_state: Dict) -> str:
        """
        Get live monitoring text update

        Args:
            current_state: Current race state

        Returns:
            Formatted text update
        """
        predictions = self.predict(current_state)

        lines = []
        lines.append("=" * 60)
        lines.append(f"LAP {current_state.get('current_lap', 0)} | P{current_state.get('current_position', 0)} | Tyre Age: {current_state.get('tyre_age', 0)} laps")
        lines.append("=" * 60)

        # Tyre wear
        if predictions['tyre_wear']:
            tw = predictions['tyre_wear']
            lines.append(f"TYRES: Next lap {tw.get('next_lap_wear', 0):.1f}% wear | {tw.get('laps_to_80_percent', 0)} laps to 80%")

        # Lap time
        if predictions['lap_time']:
            lt = predictions['lap_time']
            lines.append(f"LAP TIME: Next {lt.get('next_lap_time_str', 'N/A')} | Deg: +{lt.get('degradation_per_lap_ms', 0) / 1000:.3f}s/lap")

        # Pit stop
        if predictions['pit_stop']:
            ps = predictions['pit_stop']
            lines.append(f"PIT STOP: Lap {ps.get('optimal_pit_lap', 0)} → P{ps.get('expected_position', 0)}")

        # Race outcome
        if predictions['race_outcome']:
            ro = predictions['race_outcome']
            lines.append(f"OUTCOME: Podium {ro.get('podium_probability', 0) * 100:.0f}% | Expected P{ro.get('expected_position', 0)}")

        lines.append("=" * 60)

        return "\n".join(lines)
