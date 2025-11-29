"""
Pace Evolution Chart Component

Lap time evolution with ML forecasting.

Features:
- Actual lap times (solid line)
- Predicted lap times (dashed line)
- Confidence intervals (shaded region)
- Tyre compound markers
- Fuel-corrected pace option
- Sector time breakdown

Usage:
    from src.visualization import PaceEvolutionChart

    chart = PaceEvolutionChart()
    chart.plot_lap_times(session_id=1, driver_index=0)
    chart.add_forecast(current_state, num_laps=5)
    chart.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QCheckBox
)
from typing import Dict, List, Optional

try:
    from ..database.repositories import LapDataRepository
    from ..ml.models import LapTimeModel
    from ..analysis import PaceAnalytics
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class PaceEvolutionChart(QWidget):
    """
    Lap time evolution visualization with forecasting

    Shows lap time trends, degradation, and ML predictions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Matplotlib figure
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Data
        self.session_id = None
        self.driver_index = 0
        self.lap_time_model = None
        self.fuel_corrected = False

        # Compound colors
        self.compound_colors = {
            'SOFT': '#FF0000',
            'MEDIUM': '#FFFF00',
            'HARD': '#FFFFFF',
            'INTERMEDIATE': '#00FF00',
            'WET': '#0000FF'
        }

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup widget layout"""
        layout = QVBoxLayout()

        # Controls
        controls_layout = QHBoxLayout()

        # Driver selection
        self.driver_label = QLabel("Driver:")
        self.driver_combo = QComboBox()
        self.driver_combo.addItems([f"Driver {i}" for i in range(22)])
        self.driver_combo.currentIndexChanged.connect(self._on_driver_changed)

        # Fuel correction toggle
        self.fuel_correction_cb = QCheckBox("Fuel-Corrected Pace")
        self.fuel_correction_cb.stateChanged.connect(self._on_fuel_correction_changed)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)

        # Forecast button
        self.forecast_btn = QPushButton("Add Forecast")
        self.forecast_btn.clicked.connect(self._add_forecast)

        controls_layout.addWidget(self.driver_label)
        controls_layout.addWidget(self.driver_combo)
        controls_layout.addWidget(self.fuel_correction_cb)
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(self.forecast_btn)
        controls_layout.addStretch()

        # Canvas
        layout.addLayout(controls_layout)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def plot_lap_times(
        self,
        session_id: int,
        driver_index: int = 0,
        show_forecast: bool = False
    ):
        """
        Plot lap time evolution

        Args:
            session_id: Database session ID
            driver_index: Driver index (0-21)
            show_forecast: Whether to show ML forecast
        """
        if not DATABASE_AVAILABLE:
            self._plot_demo_data()
            return

        self.session_id = session_id
        self.driver_index = driver_index

        try:
            # Load lap data
            repo = LapDataRepository()
            lap_data = repo.get_by_session_and_driver(session_id, driver_index)

            if not lap_data:
                self._plot_no_data()
                return

            # Clear axes
            self.ax.clear()

            # Extract lap times
            lap_numbers = [lap.current_lap_num for lap in lap_data if lap.last_lap_time_ms > 0]
            lap_times = [lap.last_lap_time_ms / 1000 for lap in lap_data if lap.last_lap_time_ms > 0]  # Convert to seconds

            if not lap_times:
                self._plot_no_data()
                return

            # Apply fuel correction if enabled
            if self.fuel_corrected:
                try:
                    pace_analytics = PaceAnalytics(session_id)
                    corrected = pace_analytics.calculate_fuel_corrected_pace(driver_index)
                    if corrected and corrected.corrected_lap_times_s:
                        lap_times = corrected.corrected_lap_times_s
                except:
                    pass  # Fall back to raw times

            # Plot lap times
            self.ax.plot(
                lap_numbers,
                lap_times,
                color='#1f77b4',
                linewidth=2,
                marker='o',
                markersize=5,
                label='Lap Time',
                alpha=0.8
            )

            # Add tyre compound markers
            self._add_compound_markers(lap_data, lap_numbers, lap_times)

            # Trend line
            if len(lap_numbers) > 3:
                z = np.polyfit(lap_numbers, lap_times, 1)
                p = np.poly1d(z)
                self.ax.plot(
                    lap_numbers,
                    p(lap_numbers),
                    color='red',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.6,
                    label=f'Trend ({z[0]*1000:.1f}ms/lap)'
                )

            # Styling
            self.ax.set_xlabel('Lap Number', fontsize=12)
            self.ax.set_ylabel('Lap Time (seconds)', fontsize=12)

            title = f'Pace Evolution - Driver {driver_index}'
            if self.fuel_corrected:
                title += ' (Fuel-Corrected)'
            self.ax.set_title(title, fontsize=14, fontweight='bold')

            self.ax.set_xlim(0, max(lap_numbers) + 2)
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(loc='upper right', fontsize=9)

            # Add forecast if requested
            if show_forecast:
                self._add_forecast()

            self.canvas.draw()

        except Exception as e:
            print(f"Error plotting lap times: {e}")
            self._plot_error(str(e))

    def _add_compound_markers(self, lap_data, lap_numbers, lap_times):
        """Add tyre compound markers to chart"""
        current_compound = None
        compound_start_idx = 0

        for i, lap in enumerate(lap_data):
            if lap.last_lap_time_ms <= 0:
                continue

            compound = lap.visual_tyre_compound

            # New compound
            if compound != current_compound:
                if current_compound is not None and i > 0:
                    # Mark end of previous compound
                    color = self.compound_colors.get(current_compound, '#888888')
                    self.ax.axvspan(
                        lap_numbers[compound_start_idx],
                        lap_numbers[min(i, len(lap_numbers) - 1)],
                        alpha=0.1,
                        color=color
                    )

                current_compound = compound
                compound_start_idx = i

        # Mark final compound
        if current_compound and lap_numbers:
            color = self.compound_colors.get(current_compound, '#888888')
            self.ax.axvspan(
                lap_numbers[compound_start_idx],
                lap_numbers[-1],
                alpha=0.1,
                color=color
            )

    def add_forecast(
        self,
        current_state: Dict,
        num_laps: int = 5,
        model_path: Optional[str] = None
    ):
        """
        Add ML forecast overlay

        Args:
            current_state: Current race state
            num_laps: Number of laps to forecast
            model_path: Optional path to trained model
        """
        try:
            # Load model if not loaded
            if self.lap_time_model is None:
                if model_path:
                    from ..ml.models import LapTimeModel
                    self.lap_time_model = LapTimeModel(model_path=model_path)
                else:
                    try:
                        from ..ml.models import LapTimeModel
                        self.lap_time_model = LapTimeModel(model_path='models/lap_time_model.pkl')
                    except:
                        print("Lap time model not available")
                        return

            # Get forecast
            forecast = self.lap_time_model.forecast_lap_times(current_state, num_laps=num_laps)

            # Extract current position
            current_lap = current_state.get('lap_number', 20)
            current_time = current_state.get('current_lap_time', 85000) / 1000  # Convert to seconds

            # Build forecast points
            forecast_laps = [current_lap]
            forecast_times = [current_time]

            for f in forecast.get('forecasts', []):
                forecast_laps.append(current_lap + f['lap_ahead'])
                forecast_times.append(f['predicted_time_ms'] / 1000)

            # Plot forecast
            self.ax.plot(
                forecast_laps,
                forecast_times,
                color='purple',
                linestyle='--',
                linewidth=2,
                marker='s',
                markersize=4,
                label=f'ML Forecast (+{forecast.get("degradation_per_lap_ms", 0)/1000:.3f}s/lap)',
                alpha=0.7
            )

            # Confidence band (±0.5s)
            lower_band = [t - 0.5 for t in forecast_times]
            upper_band = [t + 0.5 for t in forecast_times]

            self.ax.fill_between(
                forecast_laps,
                lower_band,
                upper_band,
                color='purple',
                alpha=0.2,
                label='Confidence (±0.5s)'
            )

            # Update legend
            self.ax.legend(loc='upper right', fontsize=9)

            self.canvas.draw()

        except Exception as e:
            print(f"Error adding forecast: {e}")

    def _add_forecast(self):
        """Add forecast button handler"""
        if not self.session_id:
            return

        try:
            repo = LapDataRepository()
            lap_data = repo.get_by_session_and_driver(self.session_id, self.driver_index)

            if not lap_data:
                return

            # Latest lap
            valid_laps = [lap for lap in lap_data if lap.last_lap_time_ms > 0]
            if not valid_laps:
                return

            latest = valid_laps[-1]

            current_state = {
                'lap_number': latest.current_lap_num,
                'tyre_age': latest.tyre_age_laps,
                'avg_wear': 50.0,  # Default
                'fuel_remaining': 40.0,  # Default
                'current_lap_time': latest.last_lap_time_ms,
                'degradation_rate': 50  # Default ms/lap
            }

            self.add_forecast(current_state, num_laps=10)

        except Exception as e:
            print(f"Error: {e}")

    def _refresh(self):
        """Refresh chart"""
        if self.session_id:
            self.plot_lap_times(self.session_id, self.driver_index)

    def _on_driver_changed(self, index):
        """Driver selection changed"""
        self.driver_index = index
        self._refresh()

    def _on_fuel_correction_changed(self, state):
        """Fuel correction toggle changed"""
        self.fuel_corrected = (state == 2)  # Qt.Checked
        self._refresh()

    def _plot_demo_data(self):
        """Plot demo data when database unavailable"""
        self.ax.clear()

        # Demo lap times with degradation
        laps = list(range(1, 51))
        base_time = 85
        degradation = 0.05  # 50ms per lap

        lap_times = [base_time + (degradation * lap) + np.random.normal(0, 0.2) for lap in laps]

        self.ax.plot(laps, lap_times, color='#1f77b4', linewidth=2, marker='o', markersize=4, label='Lap Time')

        # Trend
        z = np.polyfit(laps, lap_times, 1)
        p = np.poly1d(z)
        self.ax.plot(laps, p(laps), 'r--', linewidth=1.5, alpha=0.6, label=f'Trend (+{z[0]*1000:.1f}ms/lap)')

        self.ax.set_xlabel('Lap Number', fontsize=12)
        self.ax.set_ylabel('Lap Time (seconds)', fontsize=12)
        self.ax.set_title('Pace Evolution (Demo Data)', fontsize=14)
        self.ax.set_xlim(0, 52)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        self.canvas.draw()

    def _plot_no_data(self):
        """Plot when no data available"""
        self.ax.clear()
        self.ax.text(
            0.5, 0.5,
            'No lap data available for this session',
            ha='center', va='center',
            transform=self.ax.transAxes,
            fontsize=14
        )
        self.canvas.draw()

    def _plot_error(self, error_msg):
        """Plot error message"""
        self.ax.clear()
        self.ax.text(
            0.5, 0.5,
            f'Error loading data:\n{error_msg}',
            ha='center', va='center',
            transform=self.ax.transAxes,
            fontsize=12,
            color='red'
        )
        self.canvas.draw()
