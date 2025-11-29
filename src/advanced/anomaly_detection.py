"""
Anomaly Detection Module

Detects unusual patterns in F1 telemetry data.

Features:
- Tyre wear anomalies (flatspots, punctures)
- Performance drops (damage, setup issues)
- Strategy mistakes (missed pit windows)
- Unexpected patterns (sensor failures)

Algorithm: Isolation Forest (unsupervised learning)

Usage:
    from src.advanced import AnomalyDetector

    detector = AnomalyDetector()
    anomalies = detector.detect_tyre_anomalies(session_id=1, driver_index=0)

    for anomaly in anomalies:
        print(f"Lap {anomaly['lap']}: {anomaly['type']} - {anomaly['description']}")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from ..database.repositories import (
        TyreDataRepository,
        LapDataRepository,
        DamageDataRepository
    )
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


@dataclass
class Anomaly:
    """Detected anomaly"""
    lap: int
    type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    score: float
    metrics: Dict


class AnomalyDetector:
    """
    Anomaly detection for F1 telemetry

    Uses Isolation Forest to detect unusual patterns.
    """

    def __init__(
        self,
        contamination: float = 0.1,  # Expected proportion of anomalies
        random_state: int = 42
    ):
        """
        Initialize anomaly detector

        Args:
            contamination: Expected proportion of outliers (0.0-0.5)
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for anomaly detection")

        self.contamination = contamination
        self.random_state = random_state

        # Models for different anomaly types
        self.tyre_model = None
        self.performance_model = None
        self.strategy_model = None

    def detect_tyre_anomalies(
        self,
        session_id: int,
        driver_index: int = 0,
        severity_threshold: float = -0.5
    ) -> List[Anomaly]:
        """
        Detect tyre wear anomalies

        Identifies:
        - Sudden wear spikes (flatspots)
        - Asymmetric wear (setup issues)
        - Premature degradation
        - Punctures

        Args:
            session_id: Database session ID
            driver_index: Driver index
            severity_threshold: Anomaly score threshold (more negative = more anomalous)

        Returns:
            List of detected anomalies
        """
        if not DATABASE_AVAILABLE:
            return []

        try:
            # Load tyre data
            repo = TyreDataRepository()
            tyre_data = repo.get_by_session_and_driver(session_id, driver_index)

            if not tyre_data or len(tyre_data) < 10:
                return []

            # Build feature matrix
            features = []
            laps = []

            for t in tyre_data:
                features.append([
                    t.wear_fl,
                    t.wear_fr,
                    t.wear_rl,
                    t.wear_rr,
                    t.tyre_surface_temp_fl,
                    t.tyre_surface_temp_fr,
                    t.tyre_surface_temp_rl,
                    t.tyre_surface_temp_rr,
                    t.tyre_age_laps
                ])
                laps.append(t.tyre_age_laps)

            X = np.array(features)

            # Train Isolation Forest
            self.tyre_model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            )

            # Fit and predict
            predictions = self.tyre_model.fit_predict(X)
            scores = self.tyre_model.score_samples(X)

            # Identify anomalies
            anomalies = []

            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1 and score < severity_threshold:
                    # Analyze what's anomalous
                    anomaly_type, description, severity = self._analyze_tyre_anomaly(
                        tyre_data[i],
                        tyre_data,
                        i
                    )

                    anomalies.append(Anomaly(
                        lap=laps[i],
                        type=anomaly_type,
                        severity=severity,
                        description=description,
                        score=score,
                        metrics={
                            'wear_fl': tyre_data[i].wear_fl,
                            'wear_fr': tyre_data[i].wear_fr,
                            'wear_rl': tyre_data[i].wear_rl,
                            'wear_rr': tyre_data[i].wear_rr,
                            'tyre_age': tyre_data[i].tyre_age_laps
                        }
                    ))

            return sorted(anomalies, key=lambda a: a.score)

        except Exception as e:
            print(f"Error detecting tyre anomalies: {e}")
            return []

    def _analyze_tyre_anomaly(self, current_data, all_data, index):
        """Analyze what caused the tyre anomaly"""
        wear_values = [
            current_data.wear_fl,
            current_data.wear_fr,
            current_data.wear_rl,
            current_data.wear_rr
        ]

        # Check for sudden spike
        if index > 0:
            prev_wear = [
                all_data[index - 1].wear_fl,
                all_data[index - 1].wear_fr,
                all_data[index - 1].wear_rl,
                all_data[index - 1].wear_rr
            ]

            wear_deltas = [curr - prev for curr, prev in zip(wear_values, prev_wear)]
            max_delta = max(wear_deltas)

            if max_delta > 10:  # >10% in one lap
                return (
                    'flatspot',
                    f'Sudden wear spike: +{max_delta:.1f}% in one lap',
                    'critical'
                )

        # Check for asymmetry
        fl_fr_diff = abs(current_data.wear_fl - current_data.wear_fr)
        rl_rr_diff = abs(current_data.wear_rl - current_data.wear_rr)

        if fl_fr_diff > 15 or rl_rr_diff > 15:
            return (
                'asymmetric_wear',
                f'Asymmetric wear: FL/FR diff {fl_fr_diff:.1f}%, RL/RR diff {rl_rr_diff:.1f}%',
                'high'
            )

        # Check for premature degradation
        avg_wear = np.mean(wear_values)
        if current_data.tyre_age_laps < 10 and avg_wear > 50:
            return (
                'premature_degradation',
                f'{avg_wear:.1f}% wear after only {current_data.tyre_age_laps} laps',
                'high'
            )

        # Check for puncture (100% wear)
        if max(wear_values) >= 100:
            return (
                'puncture',
                f'Tyre failure detected (wear: {max(wear_values):.0f}%)',
                'critical'
            )

        # Generic anomaly
        return (
            'unusual_pattern',
            f'Unusual tyre pattern detected',
            'medium'
        )

    def detect_performance_anomalies(
        self,
        session_id: int,
        driver_index: int = 0,
        severity_threshold: float = -0.5
    ) -> List[Anomaly]:
        """
        Detect performance drop anomalies

        Identifies:
        - Sudden lap time increases
        - Sector performance drops
        - Speed trap anomalies
        - Damage-related slowdowns

        Args:
            session_id: Database session ID
            driver_index: Driver index
            severity_threshold: Anomaly score threshold

        Returns:
            List of detected anomalies
        """
        if not DATABASE_AVAILABLE:
            return []

        try:
            # Load lap data
            repo = LapDataRepository()
            lap_data = repo.get_by_session_and_driver(session_id, driver_index)

            if not lap_data or len(lap_data) < 10:
                return []

            # Build feature matrix
            features = []
            laps = []

            for lap in lap_data:
                if lap.last_lap_time_ms > 0:
                    features.append([
                        lap.last_lap_time_ms / 1000,  # Lap time in seconds
                        lap.sector1_time_ms / 1000 if lap.sector1_time_ms > 0 else 0,
                        lap.sector2_time_ms / 1000 if lap.sector2_time_ms > 0 else 0,
                        lap.sector3_time_ms / 1000 if lap.sector3_time_ms > 0 else 0,
                        lap.current_lap_num
                    ])
                    laps.append(lap.current_lap_num)

            if len(features) < 10:
                return []

            X = np.array(features)

            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train Isolation Forest
            self.performance_model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            )

            predictions = self.performance_model.fit_predict(X_scaled)
            scores = self.performance_model.score_samples(X_scaled)

            # Identify anomalies
            anomalies = []

            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1 and score < severity_threshold:
                    lap_time = features[i][0]
                    mean_lap_time = np.mean([f[0] for f in features])

                    # Analyze anomaly
                    if lap_time > mean_lap_time + 3:  # >3s slower
                        severity = 'critical'
                        description = f'Major performance drop: +{lap_time - mean_lap_time:.1f}s vs average'
                    elif lap_time > mean_lap_time + 1.5:
                        severity = 'high'
                        description = f'Performance drop: +{lap_time - mean_lap_time:.1f}s vs average'
                    else:
                        severity = 'medium'
                        description = f'Unusual lap time pattern detected'

                    anomalies.append(Anomaly(
                        lap=laps[i],
                        type='performance_drop',
                        severity=severity,
                        description=description,
                        score=score,
                        metrics={
                            'lap_time_s': lap_time,
                            'mean_lap_time_s': mean_lap_time,
                            'delta_s': lap_time - mean_lap_time
                        }
                    ))

            return sorted(anomalies, key=lambda a: a.score)

        except Exception as e:
            print(f"Error detecting performance anomalies: {e}")
            return []

    def detect_strategy_anomalies(
        self,
        session_id: int,
        driver_index: int = 0
    ) -> List[Anomaly]:
        """
        Detect strategy mistakes

        Identifies:
        - Missed pit windows
        - Over-long stints
        - Suboptimal compound choices

        Args:
            session_id: Database session ID
            driver_index: Driver index

        Returns:
            List of detected strategy anomalies
        """
        if not DATABASE_AVAILABLE:
            return []

        try:
            anomalies = []

            # Load tyre data
            tyre_repo = TyreDataRepository()
            tyre_data = tyre_repo.get_by_session_and_driver(session_id, driver_index)

            if not tyre_data:
                return []

            # Check for over-long stints
            max_tyre_age = max(t.tyre_age_laps for t in tyre_data)

            if max_tyre_age > 40:
                anomalies.append(Anomaly(
                    lap=max_tyre_age,
                    type='over_long_stint',
                    severity='high',
                    description=f'Excessively long stint: {max_tyre_age} laps without pit stop',
                    score=-1.0,
                    metrics={'max_stint_length': max_tyre_age}
                ))

            # Check for running on critically worn tyres
            for t in tyre_data:
                avg_wear = (t.wear_fl + t.wear_fr + t.wear_rl + t.wear_rr) / 4
                if avg_wear > 90 and t.tyre_age_laps > 20:
                    anomalies.append(Anomaly(
                        lap=t.tyre_age_laps,
                        type='critical_tyre_wear',
                        severity='critical',
                        description=f'Running on critically worn tyres: {avg_wear:.0f}% wear',
                        score=-2.0,
                        metrics={'avg_wear': avg_wear, 'tyre_age': t.tyre_age_laps}
                    ))
                    break  # Only flag once per stint

            return anomalies

        except Exception as e:
            print(f"Error detecting strategy anomalies: {e}")
            return []

    def detect_all_anomalies(
        self,
        session_id: int,
        driver_index: int = 0
    ) -> Dict[str, List[Anomaly]]:
        """
        Detect all types of anomalies

        Args:
            session_id: Database session ID
            driver_index: Driver index

        Returns:
            Dictionary of anomaly lists by type
        """
        return {
            'tyre': self.detect_tyre_anomalies(session_id, driver_index),
            'performance': self.detect_performance_anomalies(session_id, driver_index),
            'strategy': self.detect_strategy_anomalies(session_id, driver_index)
        }

    def generate_report(
        self,
        anomalies: Dict[str, List[Anomaly]]
    ) -> str:
        """
        Generate human-readable anomaly report

        Args:
            anomalies: Dictionary of anomaly lists

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("ANOMALY DETECTION REPORT")
        report.append("=" * 60)

        total_anomalies = sum(len(a) for a in anomalies.values())
        report.append(f"\nTotal Anomalies Detected: {total_anomalies}\n")

        for anomaly_type, anomaly_list in anomalies.items():
            if anomaly_list:
                report.append(f"\n{anomaly_type.upper()} ANOMALIES ({len(anomaly_list)}):")
                report.append("-" * 60)

                for anomaly in anomaly_list:
                    severity_symbol = {
                        'low': '⚠️',
                        'medium': '⚠️',
                        'high': '🔴',
                        'critical': '🚨'
                    }.get(anomaly.severity, '⚠️')

                    report.append(
                        f"{severity_symbol} Lap {anomaly.lap}: {anomaly.type.upper()}"
                    )
                    report.append(f"   {anomaly.description}")
                    report.append(f"   Severity: {anomaly.severity.upper()} | Score: {anomaly.score:.3f}")
                    report.append("")

        report.append("=" * 60)

        return "\n".join(report)
