"""
ML Feature Builder

Transforms raw telemetry data into ML-ready features for model training and prediction.

Features:
- Extract features from database
- Create lag features (previous lap data)
- Calculate rolling statistics
- Encode categorical variables
- Normalize/scale features

Usage:
    from src.ml.features import FeatureBuilder

    builder = FeatureBuilder(session_id=1)
    features = builder.build_tyre_wear_features()
    X, y = features['X'], features['y']
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ...database import db_manager
from ...database.models import LapModel, TyreDataModel, SessionModel


class FeatureBuilder:
    """
    ML feature builder

    Transforms telemetry data into ML-ready feature matrices.
    """

    def __init__(self, session_id: int):
        """
        Initialize feature builder

        Args:
            session_id: Database session ID to extract features from
        """
        self.session_id = session_id
        self._lap_data = None
        self._tyre_data = None

    @property
    def lap_data(self) -> pd.DataFrame:
        """Lazy load lap data"""
        if self._lap_data is None:
            with db_manager.get_session() as session:
                laps = session.query(LapModel).filter_by(session_id=self.session_id).all()
                self._lap_data = pd.DataFrame([{
                    'lap_number': lap.lap_number,
                    'driver_index': lap.driver_index,
                    'driver_name': lap.driver_name,
                    'lap_time_ms': lap.lap_time_ms,
                    'sector1_time_ms': lap.sector1_time_ms,
                    'sector2_time_ms': lap.sector2_time_ms,
                    'sector3_time_ms': lap.sector3_time_ms,
                    'tyre_compound': lap.tyre_compound,
                    'tyre_age_laps': lap.tyre_age_laps,
                    'tyre_wear_fl': lap.tyre_wear_fl,
                    'tyre_wear_fr': lap.tyre_wear_fr,
                    'tyre_wear_rl': lap.tyre_wear_rl,
                    'tyre_wear_rr': lap.tyre_wear_rr,
                    'fuel_remaining_laps': lap.fuel_remaining_laps,
                    'position': lap.position,
                    'speed_trap_speed': lap.speed_trap_speed
                } for lap in laps])
        return self._lap_data

    def build_tyre_wear_features(self, driver_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Build features for tyre wear prediction model

        Args:
            driver_index: Driver to build features for

        Returns:
            Dict with 'X' (features) and 'y' (target)
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        df = df.sort_values('lap_number')

        # Calculate average tyre wear
        wear_cols = ['tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr']
        df['avg_wear'] = df[wear_cols].mean(axis=1)

        # Drop rows with missing data
        df = df.dropna(subset=['avg_wear', 'tyre_age_laps', 'lap_time_ms'])

        if len(df) < 5:
            return {'X': np.array([]), 'y': np.array([]), 'feature_names': []}

        # Features
        features = []
        feature_names = []

        # Current lap features
        features.append(df['lap_number'].values)
        feature_names.append('lap_number')

        features.append(df['tyre_age_laps'].values)
        feature_names.append('tyre_age_laps')

        features.append(df['avg_wear'].values)
        feature_names.append('current_wear')

        features.append(df['lap_time_ms'].values)
        feature_names.append('lap_time_ms')

        # Fuel load
        if 'fuel_remaining_laps' in df.columns:
            features.append(df['fuel_remaining_laps'].fillna(0).values)
            feature_names.append('fuel_remaining')

        # Compound encoding (one-hot)
        if 'tyre_compound' in df.columns:
            compounds = df['tyre_compound'].fillna('UNKNOWN')
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                features.append((compounds == compound).astype(int).values)
                feature_names.append(f'compound_{compound.lower()}')

        # Lag features (previous lap)
        df['lag_1_wear'] = df['avg_wear'].shift(1)
        df['lag_1_lap_time'] = df['lap_time_ms'].shift(1)
        features.append(df['lag_1_wear'].fillna(0).values)
        features.append(df['lag_1_lap_time'].fillna(df['lap_time_ms'].mean()).values)
        feature_names.extend(['prev_lap_wear', 'prev_lap_time'])

        # Rolling statistics (3-lap window)
        df['rolling_3_wear'] = df['avg_wear'].rolling(window=3, min_periods=1).mean()
        df['rolling_3_lap_time'] = df['lap_time_ms'].rolling(window=3, min_periods=1).mean()
        features.append(df['rolling_3_wear'].values)
        features.append(df['rolling_3_lap_time'].values)
        feature_names.extend(['rolling_3_wear', 'rolling_3_lap_time'])

        # Target: Next lap's wear
        y = df['avg_wear'].shift(-1).values

        # Remove last row (no target available)
        X = np.column_stack(features)[:-1]
        y = y[:-1]

        # Remove any remaining NaN
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'scaler_needed': True
        }

    def build_lap_time_features(self, driver_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Build features for lap time forecasting model

        Args:
            driver_index: Driver to build features for

        Returns:
            Dict with 'X' (features) and 'y' (target lap time)
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        df = df.sort_values('lap_number')
        df = df.dropna(subset=['lap_time_ms'])

        if len(df) < 5:
            return {'X': np.array([]), 'y': np.array([]), 'feature_names': []}

        features = []
        feature_names = []

        # Current lap features
        features.append(df['lap_number'].values)
        feature_names.append('lap_number')

        features.append(df['tyre_age_laps'].fillna(0).values)
        feature_names.append('tyre_age')

        # Average tyre wear
        wear_cols = ['tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr']
        df['avg_wear'] = df[wear_cols].mean(axis=1)
        features.append(df['avg_wear'].fillna(0).values)
        feature_names.append('avg_wear')

        # Fuel
        features.append(df['fuel_remaining_laps'].fillna(0).values)
        feature_names.append('fuel_remaining')

        # Previous lap times (lag features)
        for i in range(1, 4):  # Last 3 laps
            lag_col = f'lag_{i}_lap_time'
            df[lag_col] = df['lap_time_ms'].shift(i)
            features.append(df[lag_col].fillna(df['lap_time_ms'].mean()).values)
            feature_names.append(f'prev_{i}_lap_time')

        # Rolling statistics
        df['rolling_5_mean'] = df['lap_time_ms'].rolling(window=5, min_periods=1).mean()
        df['rolling_5_std'] = df['lap_time_ms'].rolling(window=5, min_periods=1).std()
        features.append(df['rolling_5_mean'].values)
        features.append(df['rolling_5_std'].fillna(0).values)
        feature_names.extend(['rolling_5_mean', 'rolling_5_std'])

        # Trend (delta from previous lap)
        df['lap_time_delta'] = df['lap_time_ms'].diff()
        features.append(df['lap_time_delta'].fillna(0).values)
        feature_names.append('lap_time_trend')

        # Target: Current lap time
        y = df['lap_time_ms'].values

        X = np.column_stack(features)

        # Remove NaN
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'scaler_needed': True
        }

    def build_pit_stop_features(self, driver_index: int = 0) -> Dict:
        """
        Build features for pit stop optimizer

        Args:
            driver_index: Driver to build features for

        Returns:
            Dict with current state features
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        df = df.sort_values('lap_number')

        if df.empty:
            return {}

        # Get current lap (latest)
        current_lap = df.iloc[-1]

        # Calculate average wear
        wear_cols = ['tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr']
        avg_wear = current_lap[wear_cols].mean()

        # Recent pace (last 3 laps)
        recent_pace = df.tail(3)['lap_time_ms'].mean()

        features = {
            'current_lap': current_lap['lap_number'],
            'position': current_lap['position'],
            'tyre_age': current_lap['tyre_age_laps'],
            'tyre_wear': avg_wear,
            'tyre_compound': current_lap['tyre_compound'],
            'fuel_remaining': current_lap['fuel_remaining_laps'],
            'recent_pace_ms': recent_pace,
            'speed_trap': current_lap['speed_trap_speed']
        }

        return features

    def build_race_outcome_features(self, driver_index: int = 0) -> Dict:
        """
        Build features for race outcome prediction

        Args:
            driver_index: Driver to build features for

        Returns:
            Dict with race state features
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()

        if df.empty:
            return {}

        # Current state
        current = df.iloc[-1]

        # Calculate pace advantage
        all_drivers_pace = self.lap_data.groupby('driver_index')['lap_time_ms'].mean()
        driver_pace = all_drivers_pace.get(driver_index, 0)
        best_pace = all_drivers_pace.min()
        pace_delta = driver_pace - best_pace if driver_pace > 0 else 0

        # Tyre advantage
        wear_cols = ['tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr']
        driver_wear = current[wear_cols].mean()

        # Consistency
        lap_times = df['lap_time_ms'].dropna()
        consistency = lap_times.std() if len(lap_times) > 1 else 0

        features = {
            'position': current['position'],
            'laps_completed': len(df),
            'pace_delta_ms': pace_delta,
            'tyre_wear': driver_wear,
            'tyre_age': current['tyre_age_laps'],
            'consistency': consistency,
            'recent_pace': df.tail(5)['lap_time_ms'].mean()
        }

        return features

    def normalize_features(self, X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """
        Normalize features using StandardScaler

        Args:
            X: Feature matrix

        Returns:
            Tuple of (normalized features, fitted scaler)
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
