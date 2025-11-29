"""
Statistical Comparisons Module

Compares performance across drivers, stints, and sessions.

Features:
- Driver vs driver comparisons
- Stint vs stint analysis
- Session vs session comparison
- Statistical significance testing

Usage:
    from src.statistics import Comparisons

    comp = Comparisons(session_id=1)
    driver_comp = comp.compare_drivers([0, 1, 2])
    significance = comp.test_significance(driver_a=0, driver_b=1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

from ..database import db_manager
from ..database.models import LapModel, SessionModel


class Comparisons:
    """
    Statistical comparison analysis

    Compares performance metrics across different entities (drivers, stints, sessions).
    """

    def __init__(self, session_id: int):
        """
        Initialize comparisons analysis

        Args:
            session_id: Database session ID to analyze
        """
        self.session_id = session_id

    def compare_drivers(self, driver_indices: List[int]) -> pd.DataFrame:
        """
        Compare performance across multiple drivers

        Args:
            driver_indices: List of driver indices to compare

        Returns:
            DataFrame with driver comparison
        """
        with db_manager.get_session() as session:
            results = []

            for driver_idx in driver_indices:
                laps = session.query(LapModel).filter_by(
                    session_id=self.session_id,
                    driver_index=driver_idx,
                    lap_invalid=False
                ).all()

                if not laps:
                    continue

                lap_times = [lap.lap_time_ms for lap in laps if lap.lap_time_ms]

                if not lap_times:
                    continue

                driver_name = laps[0].driver_name if laps else f"Driver {driver_idx}"

                results.append({
                    'driver_index': driver_idx,
                    'driver_name': driver_name,
                    'best_lap_ms': min(lap_times),
                    'average_lap_ms': np.mean(lap_times),
                    'median_lap_ms': np.median(lap_times),
                    'std_dev': np.std(lap_times),
                    'consistency_score': (1 - (np.std(lap_times) / np.mean(lap_times))) * 100 if np.mean(lap_times) > 0 else 0,
                    'laps_completed': len(lap_times)
                })

            df = pd.DataFrame(results)
            if not df.empty:
                df = df.sort_values('best_lap_ms')
                df['gap_to_best_ms'] = df['best_lap_ms'] - df['best_lap_ms'].min()

            return df

    def test_significance(self, driver_a: int, driver_b: int) -> Dict:
        """
        Test if performance difference between two drivers is statistically significant

        Args:
            driver_a: First driver index
            driver_b: Second driver index

        Returns:
            Dict with significance test results
        """
        with db_manager.get_session() as session:
            laps_a = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_a,
                lap_invalid=False
            ).all()

            laps_b = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_b,
                lap_invalid=False
            ).all()

            times_a = [lap.lap_time_ms for lap in laps_a if lap.lap_time_ms]
            times_b = [lap.lap_time_ms for lap in laps_b if lap.lap_time_ms]

            if len(times_a) < 2 or len(times_b) < 2:
                return {'error': 'Insufficient data for comparison'}

            # T-test (independent samples)
            t_stat, p_value = stats.ttest_ind(times_a, times_b)

            # Effect size (Cohen's d)
            mean_a = np.mean(times_a)
            mean_b = np.mean(times_b)
            std_a = np.std(times_a)
            std_b = np.std(times_b)

            pooled_std = np.sqrt(((len(times_a) - 1) * std_a**2 + (len(times_b) - 1) * std_b**2) / (len(times_a) + len(times_b) - 2))
            cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

            # Interpret effect size
            if abs(cohens_d) < 0.2:
                effect = "negligible"
            elif abs(cohens_d) < 0.5:
                effect = "small"
            elif abs(cohens_d) < 0.8:
                effect = "medium"
            else:
                effect = "large"

            return {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,  # 95% confidence
                'cohens_d': cohens_d,
                'effect_size': effect,
                'mean_difference_ms': mean_a - mean_b,
                'driver_a_mean': mean_a,
                'driver_b_mean': mean_b,
                'interpretation': f"Driver A is {'significantly' if p_value < 0.05 else 'not significantly'} different from Driver B (p={p_value:.4f}, effect={effect})"
            }

    def compare_stints(self, driver_index: int) -> pd.DataFrame:
        """
        Compare different stints (tyre sets) for a driver

        Args:
            driver_index: Driver to analyze

        Returns:
            DataFrame with stint comparison
        """
        with db_manager.get_session() as session:
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            df = pd.DataFrame([{
                'lap_number': lap.lap_number,
                'lap_time_ms': lap.lap_time_ms,
                'tyre_compound': lap.tyre_compound,
                'tyre_age': lap.tyre_age_laps,
                'lap_invalid': lap.lap_invalid
            } for lap in laps])

            df = df[df['lap_invalid'] == False]
            df = df.dropna(subset=['lap_time_ms', 'tyre_compound'])

            # Group by compound and analyze
            stint_comparison = df.groupby('tyre_compound').agg({
                'lap_time_ms': ['mean', 'min', 'std', 'count'],
                'tyre_age': 'max'
            }).reset_index()

            stint_comparison.columns = ['compound', 'avg_lap_time', 'best_lap_time', 'std_dev', 'laps', 'max_age']

            return stint_comparison

    def compare_sessions(self, session_ids: List[int]) -> pd.DataFrame:
        """
        Compare performance across multiple sessions

        Args:
            session_ids: List of session IDs to compare

        Returns:
            DataFrame with session comparison
        """
        with db_manager.get_session() as session:
            results = []

            for session_id in session_ids:
                session_model = session.query(SessionModel).filter_by(id=session_id).first()

                if not session_model:
                    continue

                laps = session.query(LapModel).filter_by(
                    session_id=session_id,
                    lap_invalid=False
                ).all()

                if not laps:
                    continue

                lap_times = [lap.lap_time_ms for lap in laps if lap.lap_time_ms]

                if not lap_times:
                    continue

                results.append({
                    'session_id': session_id,
                    'track': session_model.track_name,
                    'session_type': session_model.session_type,
                    'date': session_model.created_at,
                    'best_lap_ms': min(lap_times),
                    'average_lap_ms': np.mean(lap_times),
                    'total_laps': len(lap_times)
                })

            return pd.DataFrame(results)

    def head_to_head(self, driver_a: int, driver_b: int) -> Dict:
        """
        Head-to-head comparison between two drivers

        Args:
            driver_a: First driver index
            driver_b: Second driver index

        Returns:
            Dict with head-to-head statistics
        """
        with db_manager.get_session() as session:
            laps_a = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_a,
                lap_invalid=False
            ).all()

            laps_b = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_b,
                lap_invalid=False
            ).all()

            if not laps_a or not laps_b:
                return {'error': 'Insufficient data'}

            # Best lap comparison
            best_a = min([lap.lap_time_ms for lap in laps_a if lap.lap_time_ms])
            best_b = min([lap.lap_time_ms for lap in laps_b if lap.lap_time_ms])

            # Average lap comparison
            avg_a = np.mean([lap.lap_time_ms for lap in laps_a if lap.lap_time_ms])
            avg_b = np.mean([lap.lap_time_ms for lap in laps_b if lap.lap_time_ms])

            # Consistency comparison (std dev)
            std_a = np.std([lap.lap_time_ms for lap in laps_a if lap.lap_time_ms])
            std_b = np.std([lap.lap_time_ms for lap in laps_b if lap.lap_time_ms])

            driver_a_name = laps_a[0].driver_name
            driver_b_name = laps_b[0].driver_name

            return {
                'driver_a': driver_a_name,
                'driver_b': driver_b_name,
                'best_lap_winner': driver_a_name if best_a < best_b else driver_b_name,
                'best_lap_gap_ms': abs(best_a - best_b),
                'average_pace_winner': driver_a_name if avg_a < avg_b else driver_b_name,
                'average_pace_gap_ms': abs(avg_a - avg_b),
                'more_consistent': driver_a_name if std_a < std_b else driver_b_name,
                'consistency_difference': abs(std_a - std_b),
                'driver_a_stats': {'best': best_a, 'avg': avg_a, 'std': std_a},
                'driver_b_stats': {'best': best_b, 'avg': avg_b, 'std': std_b}
            }

    def percentile_rank(self, driver_index: int) -> Dict:
        """
        Calculate driver's percentile rank among all drivers

        Args:
            driver_index: Driver to analyze

        Returns:
            Dict with percentile ranks
        """
        comparison = self.compare_drivers(list(range(22)))  # All possible drivers

        if comparison.empty:
            return {'error': 'No data for comparison'}

        total_drivers = len(comparison)
        driver_row = comparison[comparison['driver_index'] == driver_index]

        if driver_row.empty:
            return {'error': 'Driver not found'}

        rank = (comparison['best_lap_ms'] < driver_row.iloc[0]['best_lap_ms']).sum() + 1
        percentile = ((total_drivers - rank + 1) / total_drivers) * 100

        return {
            'rank': rank,
            'total_drivers': total_drivers,
            'percentile': percentile,
            'interpretation': f"Faster than {100 - percentile:.1f}% of drivers"
        }
