"""
Race Outcome Prediction Model

Predicts win/podium/points probabilities using classification.

Input: Grid position, pace advantage, reliability, weather, tyre strategy
Output: Win probability, podium probability, points probability
Algorithm: Random Forest Classifier / Gradient Boosting
Value: "Win: 35%, Podium: 68%, Points: 92%"

Usage:
    from src.ml.models import RaceOutcomeModel

    model = RaceOutcomeModel()
    model.train(session_ids=[1, 2, 3])
    prediction = model.predict_outcome(current_state)
"""

import numpy as np
from typing import Dict, List, Optional
import joblib
from pathlib import Path
from dataclasses import dataclass

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..features.feature_builder import FeatureBuilder


@dataclass
class RaceOutcomePrediction:
    """Race outcome prediction result"""
    win_probability: float
    podium_probability: float
    points_probability: float
    expected_position: int
    confidence: float
    factors: Dict[str, float]


class RaceOutcomeModel:
    """
    Race outcome prediction model

    Predicts probabilities of win/podium/points finish.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize race outcome model"""
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metrics = {}

        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            if XGBOOST_AVAILABLE:
                self.model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='mlogloss'
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )

    def train(self, session_ids: List[int], test_size: float = 0.2) -> Dict:
        """
        Train model on multiple sessions

        Args:
            session_ids: List of session IDs to train on
            test_size: Test set proportion

        Returns:
            Training metrics
        """
        print(f"[RaceOutcomeModel] Training on {len(session_ids)} sessions...")

        all_X, all_y = [], []

        for session_id in session_ids:
            try:
                builder = FeatureBuilder(session_id)
                features = builder.build_race_outcome_features()

                if features['X'].shape[0] > 0:
                    all_X.append(features['X'])
                    all_y.append(features['y'])
                    if not self.feature_names:
                        self.feature_names = features['feature_names']
            except Exception as e:
                print(f"[RaceOutcomeModel] Warning: Session {session_id} failed: {e}")
                continue

        if not all_X:
            raise ValueError("No training data")

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        print(f"[RaceOutcomeModel] Total samples: {len(X)}")

        # Normalize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # Train
        print(f"[RaceOutcomeModel] Training...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_test = self.model.predict(X_test)
        y_proba_test = self.model.predict_proba(X_test)

        self.metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'samples_test': len(X_test),
            'class_distribution': np.bincount(y_train).tolist()
        }

        # Try to calculate AUC for multi-class (if applicable)
        try:
            if len(np.unique(y)) > 2:
                self.metrics['test_auc'] = roc_auc_score(
                    y_test, y_proba_test, multi_class='ovr', average='weighted'
                )
            else:
                self.metrics['test_auc'] = roc_auc_score(y_test, y_proba_test[:, 1])
        except:
            self.metrics['test_auc'] = None

        print(f"[RaceOutcomeModel] Complete!")
        print(f"  Test Accuracy: {self.metrics['test_accuracy']:.3f}")
        if self.metrics['test_auc']:
            print(f"  Test AUC: {self.metrics['test_auc']:.3f}")

        return self.metrics

    def predict_outcome(self, current_state: Dict) -> RaceOutcomePrediction:
        """
        Predict race outcome probabilities

        Args:
            current_state: Current race state with position, pace, etc.

        Returns:
            Race outcome prediction with probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Convert state to features
        features = self._state_to_features(current_state)
        X = self.scaler.transform([features])

        # Get position prediction (if regression-like output)
        # For now, we'll use a simpler approach
        current_position = current_state.get('current_position', 10)
        grid_position = current_state.get('grid_position', current_position)
        pace_advantage = current_state.get('pace_advantage', 0)  # seconds per lap

        # Calculate probabilities based on position and pace
        win_prob = self._calculate_win_probability(
            current_position, grid_position, pace_advantage, current_state
        )
        podium_prob = self._calculate_podium_probability(
            current_position, grid_position, pace_advantage, current_state
        )
        points_prob = self._calculate_points_probability(
            current_position, grid_position, pace_advantage, current_state
        )

        # Estimate expected position
        expected_position = self._estimate_expected_position(
            current_position, pace_advantage, current_state
        )

        # Calculate confidence
        confidence = self._calculate_confidence(current_state)

        # Identify key factors
        factors = self._identify_key_factors(current_state)

        return RaceOutcomePrediction(
            win_probability=win_prob,
            podium_probability=podium_prob,
            points_probability=points_prob,
            expected_position=expected_position,
            confidence=confidence,
            factors=factors
        )

    def _state_to_features(self, state: Dict) -> np.ndarray:
        """Convert state to feature vector"""
        features = [
            state.get('current_position', 10),
            state.get('grid_position', 10),
            state.get('lap_number', 0),
            state.get('total_laps', 50),
            state.get('pace_advantage', 0),
            state.get('tyre_age', 0),
            state.get('fuel_remaining', 100),
            state.get('reliability_score', 0.9),
            state.get('weather_impact', 0),
            state.get('track_position_value', 0.5)
        ]
        return np.array(features)

    def _calculate_win_probability(
        self,
        current_pos: int,
        grid_pos: int,
        pace_adv: float,
        state: Dict
    ) -> float:
        """Calculate win probability"""
        # Base probability from position
        if current_pos == 1:
            base_prob = 0.70
        elif current_pos == 2:
            base_prob = 0.20
        elif current_pos == 3:
            base_prob = 0.08
        elif current_pos <= 5:
            base_prob = 0.02
        else:
            base_prob = 0.001

        # Adjust for pace advantage (per lap)
        pace_multiplier = 1 + (pace_adv * 10)  # 0.1s/lap = 1.0 multiplier

        # Adjust for laps remaining
        laps_remaining = state.get('total_laps', 50) - state.get('lap_number', 0)
        lap_factor = min(1.0, laps_remaining / 30)  # More laps = more opportunity

        # Adjust for reliability
        reliability = state.get('reliability_score', 0.9)

        win_prob = base_prob * pace_multiplier * lap_factor * reliability

        return min(1.0, max(0.0, win_prob))

    def _calculate_podium_probability(
        self,
        current_pos: int,
        grid_pos: int,
        pace_adv: float,
        state: Dict
    ) -> float:
        """Calculate podium probability"""
        # Base probability
        if current_pos <= 3:
            base_prob = 0.85
        elif current_pos <= 5:
            base_prob = 0.50
        elif current_pos <= 8:
            base_prob = 0.20
        elif current_pos <= 10:
            base_prob = 0.10
        else:
            base_prob = 0.02

        # Adjust for pace
        pace_multiplier = 1 + (pace_adv * 8)

        # Adjust for laps remaining
        laps_remaining = state.get('total_laps', 50) - state.get('lap_number', 0)
        lap_factor = min(1.0, laps_remaining / 25)

        # Reliability
        reliability = state.get('reliability_score', 0.9)

        podium_prob = base_prob * pace_multiplier * lap_factor * reliability

        return min(1.0, max(0.0, podium_prob))

    def _calculate_points_probability(
        self,
        current_pos: int,
        grid_pos: int,
        pace_adv: float,
        state: Dict
    ) -> float:
        """Calculate points (top 10) probability"""
        # Base probability
        if current_pos <= 10:
            base_prob = 0.95
        elif current_pos <= 12:
            base_prob = 0.70
        elif current_pos <= 15:
            base_prob = 0.40
        else:
            base_prob = 0.15

        # Adjust for pace
        pace_multiplier = 1 + (pace_adv * 5)

        # Reliability
        reliability = state.get('reliability_score', 0.9)

        points_prob = base_prob * pace_multiplier * reliability

        return min(1.0, max(0.0, points_prob))

    def _estimate_expected_position(
        self,
        current_pos: int,
        pace_adv: float,
        state: Dict
    ) -> int:
        """Estimate expected final position"""
        laps_remaining = state.get('total_laps', 50) - state.get('lap_number', 0)

        # Position change from pace advantage
        # 0.1s/lap over 30 laps = ~3 seconds = ~1 position
        position_change = (pace_adv * laps_remaining) / 3

        expected_pos = current_pos - position_change

        return max(1, min(20, int(round(expected_pos))))

    def _calculate_confidence(self, state: Dict) -> float:
        """Calculate prediction confidence"""
        # Higher confidence when:
        # - More laps completed (more data)
        # - Stable position (not changing rapidly)
        # - Clear pace advantage/deficit

        lap_number = state.get('lap_number', 0)
        total_laps = state.get('total_laps', 50)

        # Confidence increases with race progress
        lap_confidence = min(1.0, lap_number / 20)  # Max confidence after 20 laps

        # Base confidence
        confidence = 0.5 + (lap_confidence * 0.4)

        return min(1.0, confidence)

    def _identify_key_factors(self, state: Dict) -> Dict[str, float]:
        """Identify key factors affecting outcome"""
        factors = {}

        # Position factor
        current_pos = state.get('current_position', 10)
        if current_pos <= 3:
            factors['track_position'] = 0.9
        elif current_pos <= 6:
            factors['track_position'] = 0.6
        else:
            factors['track_position'] = 0.3

        # Pace factor
        pace_adv = state.get('pace_advantage', 0)
        if abs(pace_adv) > 0.3:
            factors['pace'] = 0.8
        elif abs(pace_adv) > 0.1:
            factors['pace'] = 0.5
        else:
            factors['pace'] = 0.2

        # Tyre factor
        tyre_age = state.get('tyre_age', 0)
        if tyre_age < 10:
            factors['tyre_condition'] = 0.8
        elif tyre_age < 20:
            factors['tyre_condition'] = 0.5
        else:
            factors['tyre_condition'] = 0.3

        # Reliability factor
        reliability = state.get('reliability_score', 0.9)
        factors['reliability'] = reliability

        return factors

    def predict_multiple_scenarios(
        self,
        base_state: Dict,
        scenarios: List[Dict]
    ) -> List[Dict]:
        """
        Predict outcomes for multiple scenarios

        Useful for "what-if" analysis.

        Args:
            base_state: Base race state
            scenarios: List of scenario modifications

        Returns:
            List of predictions for each scenario
        """
        results = []

        for i, scenario in enumerate(scenarios):
            # Merge scenario with base state
            state = {**base_state, **scenario}

            prediction = self.predict_outcome(state)

            results.append({
                'scenario': scenario.get('name', f'Scenario {i + 1}'),
                'win_probability': round(prediction.win_probability, 3),
                'podium_probability': round(prediction.podium_probability, 3),
                'points_probability': round(prediction.points_probability, 3),
                'expected_position': prediction.expected_position,
                'confidence': round(prediction.confidence, 3)
            })

        return results

    def save(self, path: str):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }, path)

    def load(self, path: str):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.metrics = data.get('metrics', {})
