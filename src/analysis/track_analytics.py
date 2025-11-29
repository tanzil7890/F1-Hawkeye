"""
Track Analytics Module

Analyzes track evolution, grip levels, and session progression.

Features:
- Track evolution (grip improvement over session)
- Rubber buildup effect
- Optimal lap progression
- Weather impact on track conditions

Usage:
    from src.analysis import TrackAnalytics

    analytics = TrackAnalytics(session_id=1)
    evolution = analytics.analyze_track_evolution()
    grip = analytics.calculate_grip_progression()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..database import db_manager
from ..database.models import LapModel, WeatherSampleModel


@dataclass
class TrackEvolutionAnalysis:
    """Results of track evolution analysis"""
    grip_improvement_per_lap: float  # Lap time improvement per lap
    early_session_pace_ms: float  # Average of first 10 laps
    late_session_pace_ms: float  # Average of last 10 laps
    total_improvement_ms: float  # Early - Late
    improvement_percentage: float
    optimal_lap_window: Tuple[int, int]  # Best lap range


@dataclass
class WeatherImpactAnalysis:
    """Results of weather impact analysis"""
    dry_pace_ms: float
    wet_pace_ms: float
    pace_delta_ms: float  # Difference between dry and wet
    weather_changes: List[Dict]  # List of weather change events


class TrackAnalytics:
    """
    Track evolution and condition analytics

    Analyzes how track conditions improve over a session as rubber builds up.
    """

    def __init__(self, session_id: int):
        """
        Initialize track analytics for a session

        Args:
            session_id: Database session ID to analyze
        """
        self.session_id = session_id
        self._lap_data = None
        self._weather_data = None

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
                    'lap_invalid': lap.lap_invalid
                } for lap in laps])
        return self._lap_data

    @property
    def weather_data(self) -> pd.DataFrame:
        """Lazy load weather data"""
        if self._weather_data is None:
            with db_manager.get_session() as session:
                weather_samples = session.query(WeatherSampleModel).filter_by(session_id=self.session_id).all()
                self._weather_data = pd.DataFrame([{
                    'timestamp': sample.timestamp,
                    'weather': sample.weather,
                    'air_temp': sample.air_temp,
                    'track_temp': sample.track_temp,
                    'rain_percentage': sample.rain_percentage
                } for sample in weather_samples])
        return self._weather_data

    def analyze_track_evolution(self, exclude_invalid: bool = True) -> TrackEvolutionAnalysis:
        """
        Analyze how track evolves (gets faster) over the session

        Args:
            exclude_invalid: Exclude invalid laps from analysis

        Returns:
            TrackEvolutionAnalysis with track evolution metrics
        """
        df = self.lap_data.copy()

        if exclude_invalid:
            df = df[df['lap_invalid'] == False]

        df = df.dropna(subset=['lap_time_ms'])

        if df.empty:
            raise ValueError("No valid lap data available")

        # Get average lap time per lap number across all drivers
        avg_by_lap = df.groupby('lap_number')['lap_time_ms'].mean().reset_index()
        avg_by_lap = avg_by_lap.sort_values('lap_number')

        if len(avg_by_lap) < 10:
            raise ValueError("Not enough laps to analyze track evolution")

        # Early session pace (first 10 laps)
        early_laps = avg_by_lap.head(10)
        early_pace = early_laps['lap_time_ms'].mean()

        # Late session pace (last 10 laps)
        late_laps = avg_by_lap.tail(10)
        late_pace = late_laps['lap_time_ms'].mean()

        # Total improvement
        total_improvement = early_pace - late_pace
        improvement_pct = (total_improvement / early_pace) * 100 if early_pace > 0 else 0

        # Grip improvement per lap (linear regression)
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            avg_by_lap['lap_number'], avg_by_lap['lap_time_ms']
        )

        # Negative slope = track getting faster
        grip_improvement = -slope  # Make positive for "improvement"

        # Find optimal lap window (lowest average pace)
        window_size = 10
        rolling_avg = avg_by_lap['lap_time_ms'].rolling(window=window_size).mean()
        optimal_window_end = rolling_avg.idxmin()
        optimal_window_start = max(0, optimal_window_end - window_size + 1)

        return TrackEvolutionAnalysis(
            grip_improvement_per_lap=grip_improvement,
            early_session_pace_ms=early_pace,
            late_session_pace_ms=late_pace,
            total_improvement_ms=total_improvement,
            improvement_percentage=improvement_pct,
            optimal_lap_window=(int(optimal_window_start), int(optimal_window_end))
        )

    def calculate_grip_progression(self) -> pd.DataFrame:
        """
        Calculate grip progression lap by lap

        Returns:
            DataFrame with lap-by-lap grip estimate
        """
        df = self.lap_data.copy()
        df = df[df['lap_invalid'] == False]

        # Average lap time per lap number
        grip = df.groupby('lap_number').agg({
            'lap_time_ms': ['mean', 'min', 'std', 'count']
        }).reset_index()

        grip.columns = ['lap_number', 'avg_lap_time', 'fastest_lap', 'std_dev', 'cars_on_track']

        # Calculate grip improvement (inverse of lap time)
        # Faster lap time = better grip
        baseline = grip['avg_lap_time'].max()
        grip['grip_index'] = (baseline - grip['avg_lap_time']) / baseline * 100

        return grip

    def analyze_weather_impact(self) -> WeatherImpactAnalysis:
        """
        Analyze impact of weather on lap times

        Returns:
            WeatherImpactAnalysis with weather impact metrics
        """
        if self.weather_data.empty:
            return WeatherImpactAnalysis(
                dry_pace_ms=0,
                wet_pace_ms=0,
                pace_delta_ms=0,
                weather_changes=[]
            )

        df = self.lap_data.copy()
        weather = self.weather_data.copy()

        # Simplified: Assume dry if no rain, wet if rain > 0
        # (In real implementation, would join lap data with weather timestamps)

        # For now, calculate overall dry vs wet based on weather samples
        has_rain = (weather['rain_percentage'] > 0).any() if 'rain_percentage' in weather.columns else False

        if has_rain:
            # Split laps into dry and wet (simplified)
            # In production, would match lap timestamps with weather
            dry_laps = df[df['lap_invalid'] == False].head(len(df) // 2)  # Assume first half is dry
            wet_laps = df[df['lap_invalid'] == False].tail(len(df) // 2)  # Assume second half has rain

            dry_pace = dry_laps['lap_time_ms'].mean() if not dry_laps.empty else 0
            wet_pace = wet_laps['lap_time_ms'].mean() if not wet_laps.empty else 0
            delta = wet_pace - dry_pace
        else:
            dry_pace = df['lap_time_ms'].mean()
            wet_pace = 0
            delta = 0

        # Weather changes
        changes = []
        if not weather.empty:
            prev_weather = None
            for idx, row in weather.iterrows():
                if prev_weather and prev_weather != row['weather']:
                    changes.append({
                        'from': prev_weather,
                        'to': row['weather'],
                        'timestamp': row['timestamp'] if 'timestamp' in row else None
                    })
                prev_weather = row['weather']

        return WeatherImpactAnalysis(
            dry_pace_ms=dry_pace,
            wet_pace_ms=wet_pace,
            pace_delta_ms=delta,
            weather_changes=changes
        )

    def find_optimal_session_time(self) -> Dict:
        """
        Find best time in session to set a fast lap

        Returns:
            Dict with optimal session timing
        """
        try:
            evolution = self.analyze_track_evolution()

            optimal_start, optimal_end = evolution.optimal_lap_window

            return {
                'optimal_lap_range': (optimal_start, optimal_end),
                'track_improvement_total_ms': evolution.total_improvement_ms,
                'recommendation': f"Best lap window: Laps {optimal_start}-{optimal_end}"
            }
        except Exception as e:
            return {'error': str(e)}
