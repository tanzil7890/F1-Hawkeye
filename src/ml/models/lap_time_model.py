"""
Lap Time Forecasting Model

Predicts future lap times based on tyre degradation, fuel load, and track conditions.

Model: Random Forest Regressor / XGBoost
Target: Lap time in milliseconds
Accuracy Goal: ±0.5s per lap

Usage:
    from src.ml.models import LapTimeModel

    model = LapTimeModel()
    model.train(session_ids=[1, 2, 3])
    forecast = model.forecast_lap_times(current_state, num_laps=5)
"""

import numpy as np
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
from sklearn.preprocessing import StandardScaler

from ..features.feature_builder import FeatureBuilder


class LapTimeModel:
    """
    Lap time forecasting model

    Predicts future lap times with degradation effects.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize lap time model"""
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metrics = {}

        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            if XGBOOST_AVAILABLE:
                self.model = XGBRegressor(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.05,
                    random_state=42
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=12,
                    random_state=42
                )

    def train(self, session_ids: List[int], test_size: float = 0.2) -> Dict:
        """Train model on multiple sessions"""
        print(f"[LapTimeModel] Training on {len(session_ids)} sessions...")

        all_X, all_y = [], []

        for session_id in session_ids:
            try:
                builder = FeatureBuilder(session_id)
                features = builder.build_lap_time_features(driver_index=0)

                if features['X'].shape[0] > 0:
                    all_X.append(features['X'])
                    all_y.append(features['y'])
                    if not self.feature_names:
                        self.feature_names = features['feature_names']
            except Exception as e:
                print(f"[LapTimeModel] Warning: Session {session_id} failed: {e}")
                continue

        if not all_X:
            raise ValueError("No training data")

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        print(f"[LapTimeModel] Total samples: {len(X)}")

        # Normalize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # Train
        print(f"[LapTimeModel] Training...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_test = self.model.predict(X_test)

        self.metrics = {
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mae_seconds': mean_absolute_error(y_test, y_pred_test) / 1000,
            'samples_test': len(X_test)
        }

        print(f"[LapTimeModel] Complete!")
        print(f"  Test MAE: {self.metrics['test_mae_seconds']:.3f}s")
        print(f"  Test R²: {self.metrics['test_r2']:.3f}")

        return self.metrics

    def forecast_lap_times(self, current_state: Dict, num_laps: int = 5) -> Dict:
        """
        Forecast lap times for next N laps

        Args:
            current_state: Current telemetry state
            num_laps: Number of laps to forecast

        Returns:
            Dict with forecast
        """
        if self.model is None:
            raise ValueError("Model not trained")

        features = self._state_to_features(current_state)
        X = self.scaler.transform([features])

        # Predict next lap
        predicted_time = self.model.predict(X)[0]

        # Estimate degradation
        current_time = current_state.get('current_lap_time', predicted_time)
        degradation_per_lap = current_state.get('degradation_rate', 50)  # ms/lap default

        # Forecast future laps
        forecasts = []
        for i in range(1, num_laps + 1):
            forecast_time = predicted_time + (degradation_per_lap * (i - 1))
            forecasts.append({
                'lap_ahead': i,
                'predicted_time_ms': forecast_time,
                'predicted_time_str': self._format_lap_time(forecast_time)
            })

        return {
            'next_lap_time_ms': predicted_time,
            'next_lap_time_str': self._format_lap_time(predicted_time),
            'degradation_per_lap_ms': degradation_per_lap,
            'forecasts': forecasts,
            'confidence': self.metrics.get('test_r2', 0)
        }

    def _state_to_features(self, state: Dict) -> np.ndarray:
        """Convert state to feature vector"""
        features = [
            state.get('lap_number', 0),
            state.get('tyre_age', 0),
            state.get('avg_wear', 0),
            state.get('fuel_remaining', 0),
            state.get('prev_1_lap_time', state.get('current_lap_time', 0)),
            state.get('prev_2_lap_time', state.get('current_lap_time', 0)),
            state.get('prev_3_lap_time', state.get('current_lap_time', 0)),
            state.get('rolling_5_mean', state.get('current_lap_time', 0)),
            state.get('rolling_5_std', 0),
            state.get('lap_time_trend', 0)
        ]
        return np.array(features)

    def _format_lap_time(self, ms: float) -> str:
        """Format milliseconds to M:SS.sss"""
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}:{seconds:06.3f}"

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
