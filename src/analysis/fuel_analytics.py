"""
Fuel Analytics Module

Analyzes fuel consumption, stint planning, and fuel-corrected pace.

Features:
- Fuel consumption per lap
- Fuel-corrected lap times
- Stint planning and strategy
- Remaining laps calculation
- Fuel saving analysis

Usage:
    from src.analysis import FuelAnalytics

    analytics = FuelAnalytics(session_id=1)
    consumption = analytics.calculate_fuel_consumption()
    corrected_pace = analytics.get_fuel_corrected_pace()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclass

es import dataclass

from ..database import db_manager
from ..database.models import LapModel


@dataclass
class FuelConsumptionAnalysis:
    """Results of fuel consumption analysis"""
    average_fuel_per_lap: float
    total_laps_analyzed: int
    current_fuel_remaining: float
    predicted_laps_remaining: float
    fuel_saving_mode: bool  # True if driver is saving fuel
    consumption_by_lap: pd.DataFrame


@dataclass
class StintPlanningAnalysis:
    """Results of stint planning analysis"""
    optimal_stint_length: int
    fuel_load_at_start: float
    predicted_fuel_at_end: float
    safety_margin_laps: float
    recommended_fuel_load: float


FUEL_EFFECT_PER_LAP = 0.03  # Lap time improvement per lap as fuel burns (3% per 100kg typical)


class FuelAnalytics:
    """
    Fuel consumption and strategy analytics

    Analyzes fuel usage patterns, predicts remaining laps, and optimizes stint planning.
    """

    def __init__(self, session_id: int):
        """
        Initialize fuel analytics for a session

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
                    'fuel_remaining_laps': lap.fuel_remaining_laps,
                    'position': lap.position
                } for lap in laps])
        return self._lap_data

    def calculate_fuel_consumption(self, driver_index: int = 0) -> FuelConsumptionAnalysis:
        """
        Calculate fuel consumption rate

        Args:
            driver_index: Driver to analyze

        Returns:
            FuelConsumptionAnalysis with consumption metrics
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()

        if df.empty:
            raise ValueError(f"No lap data found for driver {driver_index}")

        df = df.sort_values('lap_number')
        df = df.dropna(subset=['fuel_remaining_laps'])

        if len(df) < 2:
            raise ValueError("Not enough fuel data available")

        # Calculate fuel consumption per lap (delta in fuel_remaining_laps)
        df['fuel_delta'] = -df['fuel_remaining_laps'].diff()  # Negative because fuel decreases

        # Remove outliers (pit stops show as huge jumps)
        consumption_per_lap = df[df['fuel_delta'] > 0]['fuel_delta']

        avg_fuel_per_lap = consumption_per_lap.mean() if not consumption_per_lap.empty else 0.0

        # Current fuel
        current_fuel = df.iloc[-1]['fuel_remaining_laps']

        # Predicted laps remaining
        predicted_laps = current_fuel / avg_fuel_per_lap if avg_fuel_per_lap > 0 else 0

        # Detect fuel saving mode (consumption below average)
        recent_consumption = consumption_per_lap.tail(3).mean() if len(consumption_per_lap) >= 3 else avg_fuel_per_lap
        fuel_saving = recent_consumption < (avg_fuel_per_lap * 0.9)  # 10% less than average

        # Consumption by lap
        consumption_df = df[['lap_number', 'fuel_remaining_laps', 'fuel_delta']].copy()

        return FuelConsumptionAnalysis(
            average_fuel_per_lap=avg_fuel_per_lap,
            total_laps_analyzed=len(consumption_per_lap),
            current_fuel_remaining=current_fuel,
            predicted_laps_remaining=predicted_laps,
            fuel_saving_mode=fuel_saving,
            consumption_by_lap=consumption_df
        )

    def get_fuel_corrected_pace(self, driver_index: int = 0) -> pd.DataFrame:
        """
        Calculate fuel-corrected lap times

        Heavier fuel load slows the car, so this removes fuel effect to show true pace.

        Args:
            driver_index: Driver to analyze

        Returns:
            DataFrame with fuel-corrected lap times
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        df = df.dropna(subset=['lap_time_ms', 'fuel_remaining_laps'])

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values('lap_number')

        # Estimate fuel load effect
        # Assume: More fuel remaining = slower lap times
        # Correction: Add time back based on fuel load
        max_fuel = df['fuel_remaining_laps'].max()

        # Fuel effect: ~0.03 seconds per lap per 1 lap of fuel
        df['fuel_correction_ms'] = (max_fuel - df['fuel_remaining_laps']) * FUEL_EFFECT_PER_LAP * 1000

        # Corrected lap time (what lap time would be with full tank)
        df['lap_time_corrected_ms'] = df['lap_time_ms'] + df['fuel_correction_ms']

        return df[['lap_number', 'lap_time_ms', 'lap_time_corrected_ms', 'fuel_remaining_laps', 'fuel_correction_ms']]

    def plan_stint(self, driver_index: int = 0, target_laps: int = 20, safety_margin: float = 2.0) -> StintPlanningAnalysis:
        """
        Plan fuel load for a stint

        Args:
            driver_index: Driver to analyze
            target_laps: Desired stint length
            safety_margin: Extra laps of fuel to carry

        Returns:
            StintPlanningAnalysis with fuel planning
        """
        try:
            consumption = self.calculate_fuel_consumption(driver_index)

            fuel_per_lap = consumption.average_fuel_per_lap

            # Calculate required fuel
            fuel_needed = fuel_per_lap * target_laps
            recommended_fuel = fuel_per_lap * (target_laps + safety_margin)

            # Predict end-of-stint fuel
            predicted_end_fuel = recommended_fuel - (fuel_per_lap * target_laps)

            return StintPlanningAnalysis(
                optimal_stint_length=target_laps,
                fuel_load_at_start=recommended_fuel,
                predicted_fuel_at_end=predicted_end_fuel,
                safety_margin_laps=safety_margin,
                recommended_fuel_load=recommended_fuel
            )
        except Exception as e:
            return StintPlanningAnalysis(
                optimal_stint_length=0,
                fuel_load_at_start=0.0,
                predicted_fuel_at_end=0.0,
                safety_margin_laps=safety_margin,
                recommended_fuel_load=0.0
            )

    def compare_fuel_strategies(self, driver_indices: List[int]) -> pd.DataFrame:
        """
        Compare fuel strategies across multiple drivers

        Args:
            driver_indices: List of driver indices to compare

        Returns:
            DataFrame with fuel strategy comparison
        """
        results = []

        for driver_idx in driver_indices:
            try:
                consumption = self.calculate_fuel_consumption(driver_idx)
                driver_name = self.lap_data[self.lap_data['driver_index'] == driver_idx].iloc[0]['driver_name']

                results.append({
                    'driver_index': driver_idx,
                    'driver_name': driver_name,
                    'fuel_per_lap': consumption.average_fuel_per_lap,
                    'laps_remaining': consumption.predicted_laps_remaining,
                    'fuel_saving': consumption.fuel_saving_mode
                })
            except:
                continue

        return pd.DataFrame(results)

    def detect_fuel_saving(self, driver_index: int = 0, window: int = 5) -> Dict:
        """
        Detect if driver is currently fuel saving

        Args:
            driver_index: Driver to analyze
            window: Number of recent laps to analyze

        Returns:
            Dict with fuel saving detection
        """
        try:
            consumption = self.calculate_fuel_consumption(driver_index)

            # Recent consumption vs average
            recent_laps = consumption.consumption_by_lap.tail(window)
            recent_avg = recent_laps['fuel_delta'].mean()

            overall_avg = consumption.average_fuel_per_lap

            saving_percentage = ((overall_avg - recent_avg) / overall_avg) * 100 if overall_avg > 0 else 0

            return {
                'is_saving_fuel': saving_percentage > 5,  # 5% less consumption
                'saving_percentage': saving_percentage,
                'recent_fuel_per_lap': recent_avg,
                'average_fuel_per_lap': overall_avg,
                'laps_analyzed': window
            }
        except Exception as e:
            return {'error': str(e)}
