"""
Statistical Distributions Module

Analyzes distributions of lap times and telemetry variables.

Features:
- Lap time distributions
- Normal distribution testing
- Outlier detection
- Performance percentiles

Usage:
    from src.statistics import Distributions

    dist = Distributions(session_id=1)
    lap_dist = dist.lap_time_distribution()
    outliers = dist.detect_outliers()
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats

from ..database import db_manager
from ..database.models import LapModel


class Distributions:
    """
    Statistical distribution analysis

    Analyzes distributions of performance metrics.
    """

    def __init__(self, session_id: int):
        """
        Initialize distributions analysis

        Args:
            session_id: Database session ID to analyze
        """
        self.session_id = session_id

    def lap_time_distribution(self, driver_index: int = 0) -> Dict:
        """
        Analyze lap time distribution

        Args:
            driver_index: Driver to analyze

        Returns:
            Dict with distribution statistics
        """
        with db_manager.get_session() as session:
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index,
                lap_invalid=False
            ).all()

            lap_times = [lap.lap_time_ms for lap in laps if lap.lap_time_ms]

            if len(lap_times) < 3:
                return {'error': 'Insufficient data'}

            # Basic statistics
            mean = np.mean(lap_times)
            median = np.median(lap_times)
            mode_result = stats.mode(lap_times, keepdims=True)
            mode = mode_result.mode[0] if len(mode_result.mode) > 0 else median
            std_dev = np.std(lap_times)
            variance = np.var(lap_times)

            # Percentiles
            percentiles = {
                'p25': np.percentile(lap_times, 25),
                'p50': np.percentile(lap_times, 50),
                'p75': np.percentile(lap_times, 75),
                'p90': np.percentile(lap_times, 90),
                'p95': np.percentile(lap_times, 95)
            }

            # Test for normality (Shapiro-Wilk test)
            if len(lap_times) >= 3 and len(lap_times) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(lap_times)
                is_normal = shapiro_p > 0.05  # p > 0.05 suggests normal distribution
            else:
                shapiro_stat, shapiro_p, is_normal = None, None, None

            # Skewness and kurtosis
            skewness = stats.skew(lap_times)
            kurtosis = stats.kurtosis(lap_times)

            return {
                'mean': mean,
                'median': median,
                'mode': mode,
                'std_dev': std_dev,
                'variance': variance,
                'percentiles': percentiles,
                'min': min(lap_times),
                'max': max(lap_times),
                'range': max(lap_times) - min(lap_times),
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal_distribution': is_normal,
                'shapiro_wilk_p_value': shapiro_p,
                'samples': len(lap_times)
            }

    def detect_outliers(self, driver_index: int = 0, method: str = 'iqr') -> Dict:
        """
        Detect outlier lap times

        Args:
            driver_index: Driver to analyze
            method: 'iqr' (Interquartile Range) or 'zscore' (Z-score)

        Returns:
            Dict with outlier analysis
        """
        with db_manager.get_session() as session:
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index,
                lap_invalid=False
            ).all()

            df = pd.DataFrame([{
                'lap_number': lap.lap_number,
                'lap_time_ms': lap.lap_time_ms
            } for lap in laps if lap.lap_time_ms])

            if len(df) < 3:
                return {'error': 'Insufficient data'}

            outliers = []

            if method == 'iqr':
                # Interquartile Range method
                Q1 = df['lap_time_ms'].quantile(0.25)
                Q3 = df['lap_time_ms'].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_df = df[(df['lap_time_ms'] < lower_bound) | (df['lap_time_ms'] > upper_bound)]
                outliers = outlier_df['lap_number'].tolist()

            elif method == 'zscore':
                # Z-score method (> 3 standard deviations)
                mean = df['lap_time_ms'].mean()
                std = df['lap_time_ms'].std()

                df['zscore'] = (df['lap_time_ms'] - mean) / std
                outlier_df = df[abs(df['zscore']) > 3]
                outliers = outlier_df['lap_number'].tolist()

            return {
                'outlier_laps': outliers,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'method': method,
                'total_laps': len(df)
            }

    def sector_time_distributions(self, driver_index: int = 0) -> Dict:
        """
        Analyze sector time distributions

        Args:
            driver_index: Driver to analyze

        Returns:
            Dict with sector distributions
        """
        with db_manager.get_session() as session:
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            sectors = {
                'sector1': [lap.sector1_time_ms for lap in laps if lap.sector1_time_ms],
                'sector2': [lap.sector2_time_ms for lap in laps if lap.sector2_time_ms],
                'sector3': [lap.sector3_time_ms for lap in laps if lap.sector3_time_ms]
            }

            results = {}

            for sector_name, times in sectors.items():
                if len(times) >= 3:
                    results[sector_name] = {
                        'mean': np.mean(times),
                        'median': np.median(times),
                        'std_dev': np.std(times),
                        'min': min(times),
                        'max': max(times),
                        'cv': (np.std(times) / np.mean(times)) * 100 if np.mean(times) > 0 else 0  # Coefficient of variation
                    }

            return results

    def get_performance_percentile(self, driver_index: int, target_lap_time: float) -> Dict:
        """
        Calculate what percentile a given lap time represents

        Args:
            driver_index: Driver to analyze
            target_lap_time: Lap time in ms to evaluate

        Returns:
            Dict with percentile information
        """
        with db_manager.get_session() as session:
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index,
                lap_invalid=False
            ).all()

            lap_times = [lap.lap_time_ms for lap in laps if lap.lap_time_ms]

            if not lap_times:
                return {'error': 'No lap data'}

            # Calculate percentile
            percentile = stats.percentileofscore(lap_times, target_lap_time)

            # Faster or slower than average?
            mean = np.mean(lap_times)
            delta = target_lap_time - mean

            return {
                'percentile': percentile,
                'interpretation': f"Faster than {100 - percentile:.1f}% of laps",
                'delta_from_mean_ms': delta,
                'is_above_average': target_lap_time < mean
            }
