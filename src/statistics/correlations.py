"""
Statistical Correlations Module

Calculates correlations between different telemetry variables.

Features:
- Tyre wear vs lap time correlation
- Temperature vs performance correlation
- Fuel load vs lap time correlation
- Damage impact on performance

Usage:
    from src.statistics import Correlations

    corr = Correlations(session_id=1)
    tyre_corr = corr.tyre_wear_vs_lap_time()
    temp_corr = corr.temperature_vs_performance()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

from ..database import db_manager
from ..database.models import LapModel, TyreDataModel, DamageEventModel


class Correlations:
    """
    Statistical correlation analysis

    Calculates correlations between telemetry variables to find performance relationships.
    """

    def __init__(self, session_id: int):
        """
        Initialize correlations analysis

        Args:
            session_id: Database session ID to analyze
        """
        self.session_id = session_id

    def tyre_wear_vs_lap_time(self, driver_index: int = 0) -> Dict:
        """
        Calculate correlation between tyre wear and lap time

        Args:
            driver_index: Driver to analyze

        Returns:
            Dict with correlation coefficient and p-value
        """
        with db_manager.get_session() as session:
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            df = pd.DataFrame([{
                'lap_time_ms': lap.lap_time_ms,
                'avg_wear': (lap.tyre_wear_fl + lap.tyre_wear_fr +
                           lap.tyre_wear_rl + lap.tyre_wear_rr) / 4
                           if all([lap.tyre_wear_fl, lap.tyre_wear_fr,
                                 lap.tyre_wear_rl, lap.tyre_wear_rr]) else None
            } for lap in laps])

            df = df.dropna()

            if len(df) < 3:
                return {'error': 'Insufficient data'}

            # Pearson correlation
            correlation, p_value = stats.pearsonr(df['avg_wear'], df['lap_time_ms'])

            return {
                'correlation_coefficient': correlation,
                'p_value': p_value,
                'interpretation': self._interpret_correlation(correlation),
                'samples': len(df)
            }

    def temperature_vs_performance(self, driver_index: int = 0) -> Dict:
        """
        Calculate correlation between tyre temperature and performance

        Args:
            driver_index: Driver to analyze

        Returns:
            Dict with correlation analysis
        """
        with db_manager.get_session() as session:
            # Join lap data with tyre data
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            tyres = session.query(TyreDataModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            laps_df = pd.DataFrame([{
                'lap_number': lap.lap_number,
                'lap_time_ms': lap.lap_time_ms
            } for lap in laps])

            tyres_df = pd.DataFrame([{
                'lap_number': tyre.lap_number,
                'avg_temp': (tyre.surface_temp_fl + tyre.surface_temp_fr +
                           tyre.surface_temp_rl + tyre.surface_temp_rr) / 4
            } for tyre in tyres])

            merged = pd.merge(laps_df, tyres_df, on='lap_number')
            merged = merged.dropna()

            if len(merged) < 3:
                return {'error': 'Insufficient data'}

            # Pearson correlation
            correlation, p_value = stats.pearsonr(merged['avg_temp'], merged['lap_time_ms'])

            return {
                'correlation_coefficient': correlation,
                'p_value': p_value,
                'interpretation': self._interpret_correlation(correlation),
                'optimal_temp_estimate': merged.loc[merged['lap_time_ms'].idxmin(), 'avg_temp'],
                'samples': len(merged)
            }

    def fuel_load_vs_lap_time(self, driver_index: int = 0) -> Dict:
        """
        Calculate correlation between fuel load and lap time

        Args:
            driver_index: Driver to analyze

        Returns:
            Dict with correlation analysis
        """
        with db_manager.get_session() as session:
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            df = pd.DataFrame([{
                'lap_time_ms': lap.lap_time_ms,
                'fuel_remaining': lap.fuel_remaining_laps
            } for lap in laps])

            df = df.dropna()

            if len(df) < 3:
                return {'error': 'Insufficient data'}

            # Pearson correlation
            correlation, p_value = stats.pearsonr(df['fuel_remaining'], df['lap_time_ms'])

            # Calculate fuel effect (ms per lap of fuel)
            if correlation != 0:
                # Linear regression to get slope
                slope, intercept, r_value, p_val, std_err = stats.linregress(
                    df['fuel_remaining'], df['lap_time_ms']
                )
                fuel_effect = slope  # ms per lap of fuel
            else:
                fuel_effect = 0

            return {
                'correlation_coefficient': correlation,
                'p_value': p_value,
                'fuel_effect_ms_per_lap': fuel_effect,
                'interpretation': self._interpret_correlation(correlation),
                'samples': len(df)
            }

    def damage_impact_on_performance(self, driver_index: int = 0) -> Dict:
        """
        Calculate correlation between damage and lap time loss

        Args:
            driver_index: Driver to analyze

        Returns:
            Dict with damage impact analysis
        """
        with db_manager.get_session() as session:
            damage_events = session.query(DamageEventModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            if not damage_events or not laps:
                return {'error': 'No damage data available'}

            # Create damage timeline
            damage_df = pd.DataFrame([{
                'lap_number': event.lap_number,
                'total_damage': (event.front_left_wing_damage + event.front_right_wing_damage +
                               event.rear_wing_damage + event.floor_damage)
            } for event in damage_events])

            laps_df = pd.DataFrame([{
                'lap_number': lap.lap_number,
                'lap_time_ms': lap.lap_time_ms
            } for lap in laps])

            # Merge and forward-fill damage (damage persists)
            merged = pd.merge(laps_df, damage_df, on='lap_number', how='left')
            merged['total_damage'] = merged['total_damage'].fillna(method='ffill').fillna(0)

            merged = merged[merged['total_damage'] > 0]  # Only laps with damage

            if len(merged) < 3:
                return {'error': 'Insufficient damage data'}

            # Correlation
            correlation, p_value = stats.pearsonr(merged['total_damage'], merged['lap_time_ms'])

            # Calculate damage effect (ms per % damage)
            slope, intercept, r_value, p_val, std_err = stats.linregress(
                merged['total_damage'], merged['lap_time_ms']
            )

            return {
                'correlation_coefficient': correlation,
                'p_value': p_value,
                'damage_effect_ms_per_percent': slope,
                'interpretation': self._interpret_correlation(correlation),
                'samples': len(merged)
            }

    def correlation_matrix(self, driver_index: int = 0) -> pd.DataFrame:
        """
        Calculate correlation matrix for all variables

        Args:
            driver_index: Driver to analyze

        Returns:
            DataFrame with correlation matrix
        """
        with db_manager.get_session() as session:
            laps = session.query(LapModel).filter_by(
                session_id=self.session_id,
                driver_index=driver_index
            ).all()

            df = pd.DataFrame([{
                'lap_time_ms': lap.lap_time_ms,
                'tyre_age': lap.tyre_age_laps,
                'fuel_remaining': lap.fuel_remaining_laps,
                'avg_wear': (lap.tyre_wear_fl + lap.tyre_wear_fr +
                           lap.tyre_wear_rl + lap.tyre_wear_rr) / 4
                           if all([lap.tyre_wear_fl, lap.tyre_wear_fr,
                                 lap.tyre_wear_rl, lap.tyre_wear_rr]) else None
            } for lap in laps])

            df = df.dropna()

            if len(df) < 3:
                return pd.DataFrame()

            # Calculate correlation matrix
            corr_matrix = df.corr()

            return corr_matrix

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)

        if abs_corr >= 0.9:
            strength = "Very strong"
        elif abs_corr >= 0.7:
            strength = "Strong"
        elif abs_corr >= 0.5:
            strength = "Moderate"
        elif abs_corr >= 0.3:
            strength = "Weak"
        else:
            strength = "Very weak"

        direction = "positive" if correlation > 0 else "negative"

        return f"{strength} {direction} correlation"
