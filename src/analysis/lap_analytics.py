"""
Lap Analytics Module

Analyzes lap times, calculates ideal laps, and measures consistency.

Features:
- Ideal/theoretical best lap calculation
- Consistency scoring
- Sector analysis
- Lap time distribution
- Mistake detection

Usage:
    from src.analysis import LapAnalytics

    analytics = LapAnalytics(session_id=1)
    ideal_lap = analytics.calculate_ideal_lap()
    consistency = analytics.calculate_consistency()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..database import db_manager
from ..database.models import LapModel


@dataclass
class IdealLapAnalysis:
    """Results of ideal lap calculation"""
    ideal_lap_time_ms: int
    best_actual_lap_ms: int
    improvement_potential_ms: int
    sector1_best_ms: int
    sector2_best_ms: int
    sector3_best_ms: int
    laps_analyzed: int


@dataclass
class ConsistencyAnalysis:
    """Results of consistency analysis"""
    consistency_score: float  # 0-100, higher is better
    lap_time_std_dev: float
    sector_time_std_dev: Dict[str, float]
    mistake_laps: List[int]  # Laps significantly slower than average
    clean_lap_percentage: float


class LapAnalytics:
    """
    Lap time analytics

    Analyzes lap performance, calculates ideal laps, and measures consistency.
    """

    def __init__(self, session_id: int):
        """
        Initialize lap analytics for a session

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
                    'sector1_time_ms': lap.sector1_time_ms,
                    'sector2_time_ms': lap.sector2_time_ms,
                    'sector3_time_ms': lap.sector3_time_ms,
                    'lap_invalid': lap.lap_invalid
                } for lap in laps])
        return self._lap_data

    def calculate_ideal_lap(self, driver_index: int = 0, exclude_invalid: bool = True) -> IdealLapAnalysis:
        """
        Calculate ideal/theoretical best lap time

        Combines best sector times to create the perfect lap.

        Args:
            driver_index: Driver to analyze
            exclude_invalid: Exclude invalid laps from analysis

        Returns:
            IdealLapAnalysis with ideal lap time
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()

        if exclude_invalid:
            df = df[df['lap_invalid'] == False]

        df = df.dropna(subset=['sector1_time_ms', 'sector2_time_ms', 'sector3_time_ms'])

        if df.empty:
            raise ValueError(f"No valid lap data for driver {driver_index}")

        # Find best sector times
        sector1_best = df['sector1_time_ms'].min()
        sector2_best = df['sector2_time_ms'].min()
        sector3_best = df['sector3_time_ms'].min()

        # Ideal lap = sum of best sectors
        ideal_lap = sector1_best + sector2_best + sector3_best

        # Best actual lap
        best_actual = df['lap_time_ms'].min()

        # Improvement potential
        improvement = best_actual - ideal_lap

        return IdealLapAnalysis(
            ideal_lap_time_ms=int(ideal_lap),
            best_actual_lap_ms=int(best_actual),
            improvement_potential_ms=int(improvement),
            sector1_best_ms=int(sector1_best),
            sector2_best_ms=int(sector2_best),
            sector3_best_ms=int(sector3_best),
            laps_analyzed=len(df)
        )

    def calculate_consistency(self, driver_index: int = 0, exclude_invalid: bool = True, outlier_threshold: float = 2.0) -> ConsistencyAnalysis:
        """
        Calculate driver consistency

        Args:
            driver_index: Driver to analyze
            exclude_invalid: Exclude invalid laps
            outlier_threshold: Standard deviations beyond which a lap is considered a mistake

        Returns:
            ConsistencyAnalysis with consistency metrics
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()

        if exclude_invalid:
            df = df[df['lap_invalid'] == False]

        df = df.dropna(subset=['lap_time_ms'])

        if df.empty:
            raise ValueError(f"No valid lap data for driver {driver_index}")

        # Lap time statistics
        lap_times = df['lap_time_ms']
        mean_lap = lap_times.mean()
        std_lap = lap_times.std()

        # Sector statistics
        sector_std = {}
        for sector in ['sector1_time_ms', 'sector2_time_ms', 'sector3_time_ms']:
            if sector in df.columns:
                sector_std[sector.replace('_time_ms', '')] = df[sector].std()

        # Detect mistakes (laps > mean + outlier_threshold * std_dev)
        mistake_threshold = mean_lap + (outlier_threshold * std_lap)
        mistakes = df[df['lap_time_ms'] > mistake_threshold]['lap_number'].tolist()

        # Clean lap percentage
        clean_laps = len(df) - len(mistakes)
        clean_percentage = (clean_laps / len(df)) * 100

        # Consistency score (0-100, higher is better)
        # Based on coefficient of variation (CV = std / mean)
        cv = (std_lap / mean_lap) if mean_lap > 0 else 0
        consistency_score = max(0, 100 - (cv * 1000))  # Scale CV to 0-100

        return ConsistencyAnalysis(
            consistency_score=consistency_score,
            lap_time_std_dev=std_lap,
            sector_time_std_dev=sector_std,
            mistake_laps=mistakes,
            clean_lap_percentage=clean_percentage
        )

    def get_lap_time_distribution(self, driver_index: int = 0) -> pd.DataFrame:
        """
        Get lap time distribution for visualization

        Args:
            driver_index: Driver to analyze

        Returns:
            DataFrame with lap time distribution
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        df = df.dropna(subset=['lap_time_ms'])

        if df.empty:
            return pd.DataFrame()

        # Create bins for lap time distribution
        bins = pd.cut(df['lap_time_ms'], bins=10)
        distribution = df.groupby(bins).size().reset_index(name='count')

        return distribution

    def analyze_sector_performance(self, driver_index: int = 0) -> pd.DataFrame:
        """
        Analyze sector-by-sector performance

        Args:
            driver_index: Driver to analyze

        Returns:
            DataFrame with sector analysis
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        df = df.dropna(subset=['sector1_time_ms', 'sector2_time_ms', 'sector3_time_ms'])

        if df.empty:
            return pd.DataFrame()

        sector_analysis = []

        for sector_name, sector_col in [('Sector 1', 'sector1_time_ms'),
                                         ('Sector 2', 'sector2_time_ms'),
                                         ('Sector 3', 'sector3_time_ms')]:
            sector_analysis.append({
                'sector': sector_name,
                'best_time_ms': df[sector_col].min(),
                'average_time_ms': df[sector_col].mean(),
                'worst_time_ms': df[sector_col].max(),
                'std_dev': df[sector_col].std(),
                'consistency': (df[sector_col].std() / df[sector_col].mean()) * 100 if df[sector_col].mean() > 0 else 0
            })

        return pd.DataFrame(sector_analysis)

    def compare_drivers(self, driver_indices: List[int]) -> pd.DataFrame:
        """
        Compare lap performance across multiple drivers

        Args:
            driver_indices: List of driver indices to compare

        Returns:
            DataFrame with driver comparison
        """
        results = []

        for driver_idx in driver_indices:
            try:
                ideal = self.calculate_ideal_lap(driver_idx)
                consistency = self.calculate_consistency(driver_idx)
                driver_name = self.lap_data[self.lap_data['driver_index'] == driver_idx].iloc[0]['driver_name']

                results.append({
                    'driver_index': driver_idx,
                    'driver_name': driver_name,
                    'best_lap_ms': ideal.best_actual_lap_ms,
                    'ideal_lap_ms': ideal.ideal_lap_time_ms,
                    'improvement_potential_ms': ideal.improvement_potential_ms,
                    'consistency_score': consistency.consistency_score,
                    'clean_lap_pct': consistency.clean_lap_percentage
                })
            except:
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('best_lap_ms')

        return df

    def get_lap_progression(self, driver_index: int = 0) -> pd.DataFrame:
        """
        Get lap time progression throughout the session

        Args:
            driver_index: Driver to analyze

        Returns:
            DataFrame with lap progression
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        df = df.sort_values('lap_number')
        df = df.dropna(subset=['lap_time_ms'])

        if df.empty:
            return pd.DataFrame()

        # Add rolling average
        df['rolling_avg_3'] = df['lap_time_ms'].rolling(window=3, min_periods=1).mean()
        df['rolling_avg_5'] = df['lap_time_ms'].rolling(window=5, min_periods=1).mean()

        # Add delta to best
        best_lap = df['lap_time_ms'].min()
        df['delta_to_best_ms'] = df['lap_time_ms'] - best_lap

        return df[['lap_number', 'lap_time_ms', 'rolling_avg_3', 'rolling_avg_5', 'delta_to_best_ms']]
