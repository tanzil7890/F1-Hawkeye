"""
Tyre Wear Prediction Model

Predicts future tyre wear and remaining tyre life using machine learning.

Model: Random Forest Regressor / XGBoost
Target: Tyre wear percentage at future laps
Accuracy Goal: ±3% wear prediction

Features:
- Train on historical tyre data
- Predict wear at specific lap numbers
- Estimate laps until critical wear
- Real-time wear forecasting

Usage:
    from src.ml.models import TyreWearModel

    model = TyreWearModel()
    model.train(session_ids=[1, 2, 3])
    prediction = model.predict_wear(current_state)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import joblib
from pathlib import Path

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..features.feature_builder import FeatureBuilder
from ...database import db_manager


class TyreWearModel:
    """
    Tyre wear prediction model

    Predicts tyre wear progression and remaining life.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize tyre wear model

        Args:
            model_path: Path to saved model (None = create new)
        """
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metrics = {}

        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            # Initialize new model
            if XGBOOST_AVAILABLE:
                self.model = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )

    def train(self, session_ids: List[int], test_size: float = 0.2) -> Dict:
        """
        Train model on multiple sessions

        Args:
            session_ids: List of session IDs to train on
            test_size: Fraction of data for testing

        Returns:
            Dict with training metrics
        """
        print(f"[TyreWearModel] Training on {len(session_ids)} sessions...")

        # Collect features from all sessions
        all_X = []
        all_y = []

        for session_id in session_ids:
            try:
                builder = FeatureBuilder(session_id)
                features = builder.build_tyre_wear_features(driver_index=0)

                if features['X'].shape[0] > 0:
                    all_X.append(features['X'])
                    all_y.append(features['y'])

                    if not self.feature_names:
                        self.feature_names = features['feature_names']

            except Exception as e:
                print(f"[TyreWearModel] Warning: Failed to extract features from session {session_id}: {e}")
                continue

        if not all_X:
            raise ValueError("No training data available")

        # Combine data
        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        print(f"[TyreWearModel] Total samples: {len(X)}")

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # Train model
        print(f"[TyreWearModel] Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        self.metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'samples_train': len(X_train),
            'samples_test': len(X_test)
        }

        print(f"[TyreWearModel] Training complete!")
        print(f"  Test MAE: {self.metrics['test_mae']:.2f}% wear")
        print(f"  Test RMSE: {self.metrics['test_rmse']:.2f}% wear")
        print(f"  Test R²: {self.metrics['test_r2']:.3f}")

        return self.metrics

    def predict_wear(self, current_state: Dict, future_laps: int = 10) -> Dict:
        """
        Predict tyre wear for future laps

        Args:
            current_state: Current telemetry state (from FeatureBuilder)
            future_laps: Number of laps to predict ahead

        Returns:
            Dict with wear predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Build feature vector from current state
        # This is a simplified version - in production, would use FeatureBuilder
        features = self._state_to_features(current_state)

        if features is None:
            return {'error': 'Invalid state'}

        # Normalize
        X = self.scaler.transform([features])

        # Predict next lap wear
        predicted_wear = self.model.predict(X)[0]

        # Estimate wear progression
        current_wear = current_state.get('current_wear', 0)
        wear_rate = predicted_wear - current_wear

        # Predict wear for future laps
        future_predictions = []
        for i in range(1, future_laps + 1):
            future_wear = current_wear + (wear_rate * i)
            future_predictions.append({
                'lap_ahead': i,
                'predicted_wear': min(100, max(0, future_wear))
            })

        # Calculate laps until critical wear (80%)
        if wear_rate > 0:
            laps_to_critical = (80 - current_wear) / wear_rate
        else:
            laps_to_critical = float('inf')

        return {
            'next_lap_wear': predicted_wear,
            'current_wear': current_wear,
            'wear_rate_per_lap': wear_rate,
            'laps_to_80_percent': max(0, laps_to_critical),
            'future_predictions': future_predictions,
            'confidence': self.metrics.get('test_r2', 0) if self.metrics else 0
        }

    def _state_to_features(self, state: Dict) -> Optional[np.ndarray]:
        """Convert state dict to feature vector"""
        try:
            features = [
                state.get('lap_number', 0),
                state.get('tyre_age_laps', 0),
                state.get('current_wear', 0),
                state.get('lap_time_ms', 0),
                state.get('fuel_remaining', 0),
                # Compound one-hot (simplified)
                1 if state.get('tyre_compound') == 'SOFT' else 0,
                1 if state.get('tyre_compound') == 'MEDIUM' else 0,
                1 if state.get('tyre_compound') == 'HARD' else 0,
                state.get('prev_lap_wear', state.get('current_wear', 0)),
                state.get('prev_lap_time', state.get('lap_time_ms', 0)),
                state.get('rolling_3_wear', state.get('current_wear', 0)),
                state.get('rolling_3_lap_time', state.get('lap_time_ms', 0))
            ]
            return np.array(features)
        except Exception:
            return None

    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(model_data, path)
        print(f"[TyreWearModel] Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data.get('metrics', {})
        print(f"[TyreWearModel] Model loaded from {path}")
