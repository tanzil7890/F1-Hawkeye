"""
F1 Telemetry Advanced Visualization Module

Interactive charts and graphs for telemetry analysis.

Chart Components:
- TyreWearChart: Tyre wear over time with predictions
- PaceEvolutionChart: Lap time evolution with forecasts
- PredictionOverlay: ML predictions overlaid on live data
- StrategySimulator: Compare pit strategies visually
- PerformanceHeatmap: Sector-by-sector performance analysis

Usage:
    from src.visualization import TyreWearChart, PaceEvolutionChart

    # Create tyre wear chart
    chart = TyreWearChart()
    chart.plot_wear_history(session_id=1, driver_index=0)
    chart.add_prediction(future_laps=10)

    # Create pace evolution chart
    pace_chart = PaceEvolutionChart()
    pace_chart.plot_lap_times(session_id=1)
"""

from .tyre_wear_chart import TyreWearChart
from .pace_evolution_chart import PaceEvolutionChart
from .prediction_overlay import PredictionOverlay
from .strategy_simulator import StrategySimulator
from .performance_heatmap import PerformanceHeatmap

__all__ = [
    'TyreWearChart',
    'PaceEvolutionChart',
    'PredictionOverlay',
    'StrategySimulator',
    'PerformanceHeatmap'
]
