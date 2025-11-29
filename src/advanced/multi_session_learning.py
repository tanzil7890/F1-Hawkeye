"""
Multi-Session Learning Module

Track-specific and driver-specific pattern learning across multiple sessions.

Features:
- Track-specific models (Monaco vs Spa characteristics)
- Driver-specific patterns (aggressive vs conservative)
- Weather adaptation (wet vs dry performance)
- Setup correlation analysis

Usage:
    from src.advanced import MultiSessionLearning

    learner = MultiSessionLearning()
    track_model = learner.train_track_specific_model('Monaco', session_ids=[1, 2, 3])
    driver_profile = learner.analyze_driver_patterns(driver_index=0, session_ids=[1, 2, 3])
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import joblib
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from ..database.repositories import LapDataRepository, TyreDataRepository
    from ..database import db_manager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


@dataclass
class DriverProfile:
    """Driver behavioral profile"""
    driver_index: int
    aggression_score: float  # 0-1 (conservative to aggressive)
    consistency_score: float  # 0-1 (inconsistent to consistent)
    tyre_management: str  # 'excellent', 'good', 'average', 'poor'
    pace_characteristics: Dict
    strong_tracks: List[str]
    weak_tracks: List[str]


class MultiSessionLearning:
    """
    Multi-session pattern learning

    Learns track and driver-specific patterns from historical data.
    """

    def __init__(self, models_dir: str = 'models/track_specific'):
        """Initialize multi-session learner"""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.track_models = {}
        self.driver_profiles = {}

    def train_track_specific_model(
        self,
        track_name: str,
        session_ids: List[int],
        model_type: str = 'tyre_wear'
    ) -> Dict:
        """
        Train track-specific model

        Args:
            track_name: Track name (e.g., 'Monaco', 'Spa')
            session_ids: List of session IDs for this track
            model_type: Model type ('tyre_wear', 'lap_time')

        Returns:
            Training results
        """
        if not SKLEARN_AVAILABLE or not DATABASE_AVAILABLE:
            return {'error': 'Dependencies not available'}

        print(f"[MultiSessionLearning] Training {model_type} model for {track_name}...")

        # Load data from all sessions
        all_features = []
        all_targets = []

        for session_id in session_ids:
            try:
                if model_type == 'tyre_wear':
                    features, targets = self._extract_tyre_wear_features(session_id)
                elif model_type == 'lap_time':
                    features, targets = self._extract_lap_time_features(session_id)
                else:
                    continue

                if features and targets:
                    all_features.extend(features)
                    all_targets.extend(targets)

            except Exception as e:
                print(f"  Warning: Session {session_id} failed: {e}")
                continue

        if not all_features:
            return {'error': 'No training data'}

        # Train model
        X = np.array(all_features)
        y = np.array(all_targets)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Save model
        model_key = f"{track_name}_{model_type}"
        self.track_models[model_key] = model

        model_path = self.models_dir / f"{model_key}.pkl"
        joblib.dump(model, model_path)

        results = {
            'track': track_name,
            'model_type': model_type,
            'sessions_used': len(session_ids),
            'samples': len(X),
            'train_r2': train_score,
            'test_r2': test_score,
            'model_path': str(model_path)
        }

        print(f"  ✓ Track model trained: R² = {test_score:.3f}")

        return results

    def _extract_tyre_wear_features(self, session_id: int) -> tuple:
        """Extract tyre wear features from session"""
        repo = TyreDataRepository()
        tyre_data = repo.get_by_session(session_id)

        features = []
        targets = []

        for t in tyre_data:
            # Features: tyre age, compound, temperatures
            compound_encoding = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}.get(t.visual_tyre_compound, 1)

            features.append([
                t.tyre_age_laps,
                compound_encoding,
                (t.tyre_surface_temp_fl + t.tyre_surface_temp_fr +
                 t.tyre_surface_temp_rl + t.tyre_surface_temp_rr) / 4
            ])

            # Target: average wear
            targets.append((t.wear_fl + t.wear_fr + t.wear_rl + t.wear_rr) / 4)

        return features, targets

    def _extract_lap_time_features(self, session_id: int) -> tuple:
        """Extract lap time features from session"""
        repo = LapDataRepository()
        lap_data = repo.get_by_session(session_id)

        features = []
        targets = []

        for lap in lap_data:
            if lap.last_lap_time_ms > 0:
                features.append([
                    lap.current_lap_num,
                    lap.tyre_age_laps,
                    lap.fuel_in_tank
                ])

                targets.append(lap.last_lap_time_ms / 1000)  # Convert to seconds

        return features, targets

    def analyze_driver_patterns(
        self,
        driver_index: int,
        session_ids: List[int]
    ) -> DriverProfile:
        """
        Analyze driver behavioral patterns

        Args:
            driver_index: Driver index
            session_ids: List of sessions to analyze

        Returns:
            Driver profile
        """
        if not DATABASE_AVAILABLE:
            return None

        print(f"[MultiSessionLearning] Analyzing driver {driver_index}...")

        # Collect data across sessions
        all_lap_times = []
        all_wear_rates = []
        track_performances = {}

        for session_id in session_ids:
            try:
                # Get session info
                session = db_manager.get_session_by_id(session_id)
                track_name = session.track_name if session else 'Unknown'

                # Lap times
                lap_repo = LapDataRepository()
                laps = lap_repo.get_by_session_and_driver(session_id, driver_index)

                lap_times = [lap.last_lap_time_ms / 1000 for lap in laps if lap.last_lap_time_ms > 0]

                if lap_times:
                    all_lap_times.extend(lap_times)

                    # Track performance
                    if track_name not in track_performances:
                        track_performances[track_name] = []
                    track_performances[track_name].extend(lap_times)

                # Tyre wear
                tyre_repo = TyreDataRepository()
                tyres = tyre_repo.get_by_session_and_driver(session_id, driver_index)

                for i in range(1, len(tyres)):
                    avg_wear_prev = (tyres[i-1].wear_fl + tyres[i-1].wear_fr + tyres[i-1].wear_rl + tyres[i-1].wear_rr) / 4
                    avg_wear_curr = (tyres[i].wear_fl + tyres[i].wear_fr + tyres[i].wear_rl + tyres[i].wear_rr) / 4
                    wear_rate = avg_wear_curr - avg_wear_prev

                    all_wear_rates.append(wear_rate)

            except Exception as e:
                print(f"  Warning: Session {session_id} failed: {e}")
                continue

        # Calculate profile metrics
        consistency_score = 1 - (np.std(all_lap_times) / np.mean(all_lap_times)) if all_lap_times else 0.5

        avg_wear_rate = np.mean(all_wear_rates) if all_wear_rates else 2.5

        # Tyre management classification
        if avg_wear_rate < 2.0:
            tyre_management = 'excellent'
        elif avg_wear_rate < 2.5:
            tyre_management = 'good'
        elif avg_wear_rate < 3.0:
            tyre_management = 'average'
        else:
            tyre_management = 'poor'

        # Aggression score (based on wear rate and lap time variance)
        aggression_score = min(1.0, avg_wear_rate / 5.0)

        # Strong/weak tracks
        track_avg_times = {track: np.mean(times) for track, times in track_performances.items()}

        sorted_tracks = sorted(track_avg_times.items(), key=lambda x: x[1])
        strong_tracks = [t[0] for t in sorted_tracks[:3]]
        weak_tracks = [t[0] for t in sorted_tracks[-3:]]

        profile = DriverProfile(
            driver_index=driver_index,
            aggression_score=aggression_score,
            consistency_score=consistency_score,
            tyre_management=tyre_management,
            pace_characteristics={
                'avg_lap_time': np.mean(all_lap_times) if all_lap_times else 0,
                'std_lap_time': np.std(all_lap_times) if all_lap_times else 0,
                'avg_wear_rate': avg_wear_rate
            },
            strong_tracks=strong_tracks,
            weak_tracks=weak_tracks
        )

        self.driver_profiles[driver_index] = profile

        print(f"  ✓ Driver profile created")
        print(f"    Aggression: {aggression_score:.2f}")
        print(f"    Consistency: {consistency_score:.2f}")
        print(f"    Tyre Management: {tyre_management}")

        return profile

    def load_track_model(self, track_name: str, model_type: str = 'tyre_wear'):
        """Load saved track-specific model"""
        model_key = f"{track_name}_{model_type}"
        model_path = self.models_dir / f"{model_key}.pkl"

        if model_path.exists():
            self.track_models[model_key] = joblib.load(model_path)
            return True
        return False

    def predict_with_track_model(
        self,
        track_name: str,
        features: np.ndarray,
        model_type: str = 'tyre_wear'
    ) -> np.ndarray:
        """Make prediction using track-specific model"""
        model_key = f"{track_name}_{model_type}"

        if model_key not in self.track_models:
            if not self.load_track_model(track_name, model_type):
                raise ValueError(f"No model found for {model_key}")

        model = self.track_models[model_key]
        return model.predict(features)
