"""
Pace Analytics Module

Analyzes race pace, tyre-corrected pace, and fuel-corrected pace.

Features:
- Race pace calculation
- Tyre-corrected pace
- Fuel-corrected pace
- Pace comparison across drivers
- Long-run pace simulation

Usage:
    from src.analysis import PaceAnalytics

    analytics = PaceAnalytics(session_id=1)
    race_pace = analytics.calculate_race_pace()
    corrected = analytics.get_fully_corrected_pace()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..database import db_manager
from ..database.models import LapModel


@dataclass
class RacePaceAnalysis:
    """Results of race pace analysis"""
    average_pace_ms: float
    median_pace_ms: float
    pace_std_dev: float
    fastest_3_lap_avg: float
    long_run_pace: float  # Average of laps 10+
    tyre_corrected_pace: float
    fuel_corrected_pace: float


TYRE_EFFECT_PER_PERCENT = 0.5  # ms per 1% tyre wear
FUEL_EFFECT_PER_LAP = 30  # ms per lap of fuel (heavier = slower)


class PaceAnalytics:
    """
    Race pace analytics

    Analyzes true race pace with corrections for tyre and fuel effects.
    """

    def __init__(self, session_id: int):
        """
        Initialize pace analytics for a session

        Args:
            session_id: Database session ID to analyze
        """
        self.session_id = session_id
        self._lap_data = None

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
                    'tyre_compound': lap.tyre_compound,
                    'tyre_age_laps': lap.tyre_age_laps,
                    'tyre_wear_fl': lap.tyre_wear_fl,
                    'tyre_wear_fr': lap.tyre_wear_fr,
                    'tyre_wear_rl': lap.tyre_wear_rl,
                    'tyre_wear_rr': lap.tyre_wear_rr,
                    'fuel_remaining_laps': lap.fuel_remaining_laps,
                    'lap_invalid': lap.lap_invalid
                } for lap in laps])
        return self._lap_data

    def calculate_race_pace(self, driver_index: int = 0, exclude_invalid: bool = True) -> RacePaceAnalysis:
        """
        Calculate overall race pace

        Args:
            driver_index: Driver to analyze
            exclude_invalid: Exclude invalid laps

        Returns:
            RacePaceAnalysis with pace metrics
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()

        if exclude_invalid:
            df = df[df['lap_invalid'] == False]

        df = df.dropna(subset=['lap_time_ms'])

        if df.empty:
            raise ValueError(f"No valid lap data for driver {driver_index}")

        df = df.sort_values('lap_number')

        # Basic pace metrics
        avg_pace = df['lap_time_ms'].mean()
        median_pace = df['lap_time_ms'].median()
        pace_std = df['lap_time_ms'].std()

        # Fastest 3 lap average (qualifying simulation)
        fastest_3 = df.nsmallest(3, 'lap_time_ms')['lap_time_ms'].mean() if len(df) >= 3 else avg_pace

        # Long run pace (laps 10+, representative of race pace)
        long_run = df[df['lap_number'] >= 10]['lap_time_ms'].mean() if len(df) >= 10 else avg_pace

        # Tyre-corrected pace
        tyre_corrected = self._calculate_tyre_corrected_pace(df)

        # Fuel-corrected pace
        fuel_corrected = self._calculate_fuel_corrected_pace(df)

        return RacePaceAnalysis(
            average_pace_ms=avg_pace,
            median_pace_ms=median_pace,
            pace_std_dev=pace_std,
            fastest_3_lap_avg=fastest_3,
            long_run_pace=long_run,
            tyre_corrected_pace=tyre_corrected,
            fuel_corrected_pace=fuel_corrected
        )

    def _calculate_tyre_corrected_pace(self, df: pd.DataFrame) -> float:
        """Calculate pace corrected for tyre wear"""
        wear_cols = ['tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr']

        # Average tyre wear
        df_copy = df.copy()
        df_copy['avg_wear'] = df_copy[wear_cols].mean(axis=1)

        # Correction: Remove tyre wear effect from lap time
        # Higher wear = slower lap, so subtract the wear effect
        df_copy['tyre_correction'] = df_copy['avg_wear'] * TYRE_EFFECT_PER_PERCENT
        df_copy['corrected_lap'] = df_copy['lap_time_ms'] - df_copy['tyre_correction']

        return df_copy['corrected_lap'].mean()

    def _calculate_fuel_corrected_pace(self, df: pd.DataFrame) -> float:
        """Calculate pace corrected for fuel load"""
        df_copy = df.copy()

        if 'fuel_remaining_laps' not in df_copy.columns:
            return df_copy['lap_time_ms'].mean()

        # More fuel = slower lap
        # Correction: What would lap time be with minimum fuel?
        min_fuel = df_copy['fuel_remaining_laps'].min()
        df_copy['fuel_correction'] = (df_copy['fuel_remaining_laps'] - min_fuel) * FUEL_EFFECT_PER_LAP
        df_copy['corrected_lap'] = df_copy['lap_time_ms'] - df_copy['fuel_correction']

        return df_copy['corrected_lap'].mean()

    def get_fully_corrected_pace(self, driver_index: int = 0) -> pd.DataFrame:
        """
        Get lap-by-lap pace with all corrections applied

        Args:
            driver_index: Driver to analyze

        Returns:
            DataFrame with corrected lap times
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        df = df.dropna(subset=['lap_time_ms'])

        if df.empty:
            return pd.DataFrame()

        wear_cols = ['tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr']

        # Tyre correction
        df['avg_wear'] = df[wear_cols].mean(axis=1)
        df['tyre_correction_ms'] = df['avg_wear'] * TYRE_EFFECT_PER_PERCENT

        # Fuel correction
        if 'fuel_remaining_laps' in df.columns:
            min_fuel = df['fuel_remaining_laps'].min()
            df['fuel_correction_ms'] = (df['fuel_remaining_laps'] - min_fuel) * FUEL_EFFECT_PER_LAP
        else:
            df['fuel_correction_ms'] = 0

        # Fully corrected lap time
        df['lap_time_corrected_ms'] = df['lap_time_ms'] - df['tyre_correction_ms'] - df['fuel_correction_ms']

        return df[['lap_number', 'lap_time_ms', 'lap_time_corrected_ms', 'tyre_correction_ms', 'fuel_correction_ms', 'avg_wear']]

    def compare_pace(self, driver_indices: List[int]) -> pd.DataFrame:
        """
        Compare race pace across multiple drivers

        Args:
            driver_indices: List of driver indices to compare

        Returns:
            DataFrame with pace comparison
        """
        results = []

        for driver_idx in driver_indices:
            try:
                pace = self.calculate_race_pace(driver_idx)
                driver_name = self.lap_data[self.lap_data['driver_index'] == driver_idx].iloc[0]['driver_name']

                results.append({
                    'driver_index': driver_idx,
                    'driver_name': driver_name,
                    'average_pace_ms': pace.average_pace_ms,
                    'long_run_pace_ms': pace.long_run_pace,
                    'tyre_corrected_ms': pace.tyre_corrected_pace,
                    'fuel_corrected_ms': pace.fuel_corrected_pace,
                    'fastest_3_avg_ms': pace.fastest_3_lap_avg
                })
            except:
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('average_pace_ms')

        return df

    def predict_race_time(self, driver_index: int = 0, total_race_laps: int = 58) -> Dict:
        """
        Predict total race time based on current pace

        Args:
            driver_index: Driver to analyze
            total_race_laps: Number of laps in the race

        Returns:
            Dict with race time prediction
        """
        try:
            pace = self.calculate_race_pace(driver_index)

            # Use long-run pace if available, otherwise average
            representative_pace = pace.long_run_pace if pace.long_run_pace > 0 else pace.average_pace_ms

            # Predict total race time
            predicted_total_ms = representative_pace * total_race_laps

            # Convert to minutes
            predicted_minutes = predicted_total_ms / 60000

            return {
                'predicted_race_time_ms': predicted_total_ms,
                'predicted_race_time_minutes': predicted_minutes,
                'pace_per_lap_ms': representative_pace,
                'total_laps': total_race_laps
            }
        except Exception as e:
            return {'error': str(e)}
