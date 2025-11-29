"""
Tyre Analytics Module

Analyzes tyre performance, wear rates, degradation curves, and temperature impacts.

Features:
- Wear rate per lap calculation
- Compound-specific degradation curves
- Temperature impact on wear
- Optimal temperature range detection
- Tyre life prediction

Usage:
    from src.analysis import TyreAnalytics

    analytics = TyreAnalytics(session_id=1)
    wear_rate = analytics.calculate_wear_rate()
    degradation = analytics.get_degradation_curve()
    optimal_temp = analytics.find_optimal_temperature()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..database import db_manager
from ..database.models import LapModel, TyreDataModel


@dataclass
class TyreWearAnalysis:
    """Results of tyre wear analysis"""
    compound: str
    average_wear_per_lap: float
    total_wear: float
    laps_analyzed: int
    wear_by_corner: Dict[str, float]  # FL, FR, RL, RR
    predicted_life_laps: float
    degradation_rate: float  # % per lap


@dataclass
class TyreTempAnalysis:
    """Results of tyre temperature analysis"""
    compound: str
    optimal_temp_range: Tuple[float, float]  # (min, max)
    average_performance_by_temp: Dict[str, float]  # temp_range: lap_time_delta
    overheating_laps: List[int]
    underheating_laps: List[int]


class TyreAnalytics:
    """
    Tyre performance analytics

    Analyzes tyre wear, degradation, temperature impact, and predicts tyre life.
    """

    def __init__(self, session_id: int):
        """
        Initialize tyre analytics for a session

        Args:
            session_id: Database session ID to analyze
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
                    'tyre_compound': lap.tyre_compound,
                    'tyre_age_laps': lap.tyre_age_laps,
                    'tyre_wear_fl': lap.tyre_wear_fl,
                    'tyre_wear_fr': lap.tyre_wear_fr,
                    'tyre_wear_rl': lap.tyre_wear_rl,
                    'tyre_wear_rr': lap.tyre_wear_rr
                } for lap in laps])
        return self._lap_data

    @property
    def tyre_data(self) -> pd.DataFrame:
        """Lazy load tyre data"""
        if self._tyre_data is None:
            with db_manager.get_session() as session:
                tyres = session.query(TyreDataModel).filter_by(session_id=self.session_id).all()
                self._tyre_data = pd.DataFrame([{
                    'lap_number': tyre.lap_number,
                    'driver_index': tyre.driver_index,
                    'compound': tyre.compound,
                    'age_laps': tyre.age_laps,
                    'wear_fl': tyre.wear_fl,
                    'wear_fr': tyre.wear_fr,
                    'wear_rl': tyre.wear_rl,
                    'wear_rr': tyre.wear_rr,
                    'surface_temp_fl': tyre.surface_temp_fl,
                    'surface_temp_fr': tyre.surface_temp_fr,
                    'surface_temp_rl': tyre.surface_temp_rl,
                    'surface_temp_rr': tyre.surface_temp_rr
                } for tyre in tyres])
        return self._tyre_data

    def calculate_wear_rate(self, driver_index: int = 0, compound: Optional[str] = None) -> TyreWearAnalysis:
        """
        Calculate tyre wear rate per lap

        Args:
            driver_index: Driver to analyze (default: player)
            compound: Tyre compound to analyze (None = most recent stint)

        Returns:
            TyreWearAnalysis with wear rates and predictions
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()

        if df.empty:
            raise ValueError(f"No lap data found for driver {driver_index}")

        # Filter by compound if specified
        if compound:
            df = df[df['tyre_compound'] == compound]
        else:
            # Use most recent compound
            compound = df.iloc[-1]['tyre_compound']
            df = df[df['tyre_compound'] == compound]

        if len(df) < 2:
            raise ValueError(f"Not enough laps for compound {compound}")

        # Calculate wear deltas
        df = df.sort_values('lap_number')
        wear_cols = ['tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr']

        # Drop rows with missing wear data
        df = df.dropna(subset=wear_cols)

        if len(df) < 2:
            raise ValueError("Not enough wear data available")

        # Calculate wear per lap for each corner
        wear_deltas = {}
        for col in wear_cols:
            deltas = df[col].diff().dropna()
            wear_deltas[col.replace('tyre_wear_', '').upper()] = deltas.mean()

        # Average wear across all corners
        avg_wear_per_lap = np.mean(list(wear_deltas.values()))

        # Total wear so far
        total_wear = df[wear_cols].iloc[-1].mean()

        # Predict tyre life (100% wear = dead tyre)
        predicted_life = (100 - total_wear) / avg_wear_per_lap if avg_wear_per_lap > 0 else float('inf')

        # Degradation rate (how fast wear is accelerating)
        if len(df) >= 5:
            recent_wear = df[wear_cols].iloc[-3:].mean(axis=1).diff().mean()
            early_wear = df[wear_cols].iloc[:3].mean(axis=1).diff().mean()
            degradation_rate = (recent_wear - early_wear) / early_wear if early_wear > 0 else 0
        else:
            degradation_rate = 0.0

        return TyreWearAnalysis(
            compound=compound,
            average_wear_per_lap=avg_wear_per_lap,
            total_wear=total_wear,
            laps_analyzed=len(df),
            wear_by_corner=wear_deltas,
            predicted_life_laps=predicted_life,
            degradation_rate=degradation_rate
        )

    def get_degradation_curve(self, driver_index: int = 0, compound: Optional[str] = None) -> pd.DataFrame:
        """
        Get tyre degradation curve (lap time vs tyre age)

        Args:
            driver_index: Driver to analyze
            compound: Tyre compound (None = most recent)

        Returns:
            DataFrame with columns: tyre_age_laps, lap_time_ms, wear_avg
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()

        if compound:
            df = df[df['tyre_compound'] == compound]
        else:
            compound = df.iloc[-1]['tyre_compound']
            df = df[df['tyre_compound'] == compound]

        # Calculate average wear
        wear_cols = ['tyre_wear_fl', 'tyre_wear_fr', 'tyre_wear_rl', 'tyre_wear_rr']
        df['wear_avg'] = df[wear_cols].mean(axis=1)

        # Group by tyre age and get average lap time
        curve = df.groupby('tyre_age_laps').agg({
            'lap_time_ms': 'mean',
            'wear_avg': 'mean'
        }).reset_index()

        return curve

    def find_optimal_temperature(self, driver_index: int = 0, compound: Optional[str] = None) -> TyreTempAnalysis:
        """
        Find optimal tyre temperature range for performance

        Args:
            driver_index: Driver to analyze
            compound: Tyre compound

        Returns:
            TyreTempAnalysis with optimal temperature range
        """
        # Merge lap data with tyre temperature data
        laps = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()
        tyres = self.tyre_data[self.tyre_data['driver_index'] == driver_index].copy()

        if compound:
            laps = laps[laps['tyre_compound'] == compound]
            tyres = tyres[tyres['compound'] == compound]
        else:
            compound = laps.iloc[-1]['tyre_compound'] if not laps.empty else "UNKNOWN"

        if laps.empty or tyres.empty:
            # Return default analysis
            return TyreTempAnalysis(
                compound=compound,
                optimal_temp_range=(85.0, 95.0),  # Default F1 optimal range
                average_performance_by_temp={},
                overheating_laps=[],
                underheating_laps=[]
            )

        # Merge on lap_number
        merged = pd.merge(laps, tyres, on=['lap_number', 'driver_index'], how='inner')

        # Calculate average tyre temperature
        temp_cols = ['surface_temp_fl', 'surface_temp_fr', 'surface_temp_rl', 'surface_temp_rr']
        merged['avg_temp'] = merged[temp_cols].mean(axis=1)

        # Find temperature range with best lap times
        merged = merged.sort_values('avg_temp')
        temp_bins = pd.cut(merged['avg_temp'], bins=5)
        perf_by_temp = merged.groupby(temp_bins)['lap_time_ms'].mean()

        # Find optimal bin
        optimal_bin = perf_by_temp.idxmin()
        optimal_range = (optimal_bin.left, optimal_bin.right)

        # Detect overheating (> 100°C) and underheating (< 70°C)
        overheating = merged[merged['avg_temp'] > 100]['lap_number'].tolist()
        underheating = merged[merged['avg_temp'] < 70]['lap_number'].tolist()

        # Performance by temperature range
        perf_dict = {f"{int(k.left)}-{int(k.right)}°C": v for k, v in perf_by_temp.items()}

        return TyreTempAnalysis(
            compound=compound,
            optimal_temp_range=optimal_range,
            average_performance_by_temp=perf_dict,
            overheating_laps=overheating,
            underheating_laps=underheating
        )

    def compare_compounds(self, driver_index: int = 0) -> pd.DataFrame:
        """
        Compare performance across different tyre compounds

        Args:
            driver_index: Driver to analyze

        Returns:
            DataFrame with compound comparison metrics
        """
        df = self.lap_data[self.lap_data['driver_index'] == driver_index].copy()

        comparison = df.groupby('tyre_compound').agg({
            'lap_time_ms': ['mean', 'min', 'std'],
            'tyre_age_laps': 'max',
            'lap_number': 'count'
        }).reset_index()

        comparison.columns = ['compound', 'avg_lap_time', 'best_lap_time', 'consistency', 'max_age', 'laps_completed']

        return comparison

    def predict_pitstop_window(self, driver_index: int = 0, target_wear: float = 80.0) -> Dict:
        """
        Predict when to pit based on tyre wear

        Args:
            driver_index: Driver to analyze
            target_wear: Maximum acceptable wear % before pit

        Returns:
            Dict with pit window prediction
        """
        try:
            analysis = self.calculate_wear_rate(driver_index)

            current_wear = analysis.total_wear
            wear_per_lap = analysis.average_wear_per_lap

            if wear_per_lap <= 0:
                return {
                    'recommended_pit_in_laps': float('inf'),
                    'current_wear': current_wear,
                    'target_wear': target_wear,
                    'compound': analysis.compound
                }

            laps_to_target = (target_wear - current_wear) / wear_per_lap

            return {
                'recommended_pit_in_laps': int(max(0, laps_to_target)),
                'current_wear': current_wear,
                'target_wear': target_wear,
                'wear_per_lap': wear_per_lap,
                'compound': analysis.compound,
                'predicted_life_remaining': analysis.predicted_life_laps
            }
        except Exception as e:
            return {'error': str(e)}
