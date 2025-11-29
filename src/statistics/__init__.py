"""
F1 Telemetry Statistics Module

Statistical analysis tools for performance metrics.

Statistics Modules:
- Correlations: Tyre wear vs lap time, temperature vs performance correlations
- Distributions: Lap time distributions, outlier detection, percentiles
- Comparisons: Driver vs driver, stint vs stint, statistical significance testing

Usage:
    from src.statistics import Correlations, Distributions, Comparisons

    # Correlation analysis
    corr = Correlations(session_id=1)
    tyre_corr = corr.tyre_wear_vs_lap_time()
    temp_corr = corr.temperature_vs_performance()

    # Distribution analysis
    dist = Distributions(session_id=1)
    lap_dist = dist.lap_time_distribution()
    outliers = dist.detect_outliers()

    # Comparison analysis
    comp = Comparisons(session_id=1)
    driver_comp = comp.compare_drivers([0, 1, 2])
    significance = comp.test_significance(driver_a=0, driver_b=1)
"""

from .correlations import Correlations
from .distributions import Distributions
from .comparisons import Comparisons

__all__ = [
    'Correlations',
    'Distributions',
    'Comparisons'
]
