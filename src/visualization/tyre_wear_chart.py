"""
Tyre Wear Chart Component

Interactive tyre wear visualization with ML predictions.

Features:
- Historical wear curves for all compounds
- Predicted wear with confidence bands
- Critical wear threshold indicators (80%, 100%)
- Compound comparison overlay
- Interactive legend and tooltips

Usage:
    from src.visualization import TyreWearChart

    chart = TyreWearChart()
    chart.plot_wear_history(session_id=1, driver_index=0)
    chart.add_prediction(current_state, future_laps=10)
    chart.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox
from typing import Dict, List, Optional

try:
    from ..database.repositories import TyreDataRepository, LapDataRepository
    from ..ml.models import TyreWearModel
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class TyreWearChart(QWidget):
    """
    Tyre wear visualization with predictions

    Embeds Matplotlib chart in Qt widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Data
        self.session_id = None
        self.driver_index = 0
        self.tyre_model = None

        # Compound colors
        self.compound_colors = {
            'SOFT': '#FF0000',      # Red
            'MEDIUM': '#FFFF00',    # Yellow
            'HARD': '#FFFFFF',      # White
            'INTERMEDIATE': '#00FF00',  # Green
            'WET': '#0000FF'        # Blue
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

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)

        # Add prediction button
        self.predict_btn = QPushButton("Add Prediction")
        self.predict_btn.clicked.connect(self._add_prediction)

        controls_layout.addWidget(self.driver_label)
        controls_layout.addWidget(self.driver_combo)
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(self.predict_btn)
        controls_layout.addStretch()

        # Canvas
        layout.addLayout(controls_layout)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def plot_wear_history(
        self,
        session_id: int,
        driver_index: int = 0,
        show_prediction: bool = False
    ):
        """
        Plot historical tyre wear

        Args:
            session_id: Database session ID
            driver_index: Driver index (0-21)
            show_prediction: Whether to show ML predictions
        """
        if not DATABASE_AVAILABLE:
            self._plot_demo_data()
            return

        self.session_id = session_id
        self.driver_index = driver_index

        try:
            # Load tyre data
            repo = TyreDataRepository()
            tyre_data = repo.get_by_session_and_driver(session_id, driver_index)

            if not tyre_data:
                self._plot_no_data()
                return

            # Clear axes
            self.ax.clear()

            # Group by tyre age (each stint)
            stints = self._group_into_stints(tyre_data)

            # Plot each stint
            for stint_idx, stint in enumerate(stints):
                laps = [t.tyre_age_laps for t in stint]
                wear_fl = [t.wear_fl for t in stint]
                wear_fr = [t.wear_fr for t in stint]
                wear_rl = [t.wear_rl for t in stint]
                wear_rr = [t.wear_rr for t in stint]

                # Average wear
                avg_wear = [(fl + fr + rl + rr) / 4 for fl, fr, rl, rr in zip(wear_fl, wear_fr, wear_rl, wear_rr)]

                # Compound color
                compound = stint[0].visual_tyre_compound if stint else 'MEDIUM'
                color = self.compound_colors.get(compound, '#888888')

                # Plot wear curve
                self.ax.plot(
                    laps, avg_wear,
                    color=color,
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label=f'Stint {stint_idx + 1} ({compound})',
                    alpha=0.8
                )

                # Individual tyre traces (lighter)
                self.ax.plot(laps, wear_fl, color=color, linestyle='--', alpha=0.3, linewidth=1)
                self.ax.plot(laps, wear_fr, color=color, linestyle='--', alpha=0.3, linewidth=1)
                self.ax.plot(laps, wear_rl, color=color, linestyle='--', alpha=0.3, linewidth=1)
                self.ax.plot(laps, wear_rr, color=color, linestyle='--', alpha=0.3, linewidth=1)

            # Critical thresholds
            max_lap = max([t.tyre_age_laps for stint in stints for t in stint]) if stints else 50
            self.ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% Wear (Caution)')
            self.ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Wear (Critical)')

            # Styling
            self.ax.set_xlabel('Tyre Age (Laps)', fontsize=12)
            self.ax.set_ylabel('Tyre Wear (%)', fontsize=12)
            self.ax.set_title(f'Tyre Wear Evolution - Driver {driver_index}', fontsize=14, fontweight='bold')
            self.ax.set_xlim(0, max_lap + 5)
            self.ax.set_ylim(0, 120)
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(loc='upper left', fontsize=9)

            # Add prediction if requested
            if show_prediction:
                self._add_prediction()

            self.canvas.draw()

        except Exception as e:
            print(f"Error plotting tyre wear: {e}")
            self._plot_error(str(e))

    def _group_into_stints(self, tyre_data):
        """Group tyre data into stints"""
        if not tyre_data:
            return []

        stints = []
        current_stint = [tyre_data[0]]

        for i in range(1, len(tyre_data)):
            # New stint if tyre age resets or compound changes
            if (tyre_data[i].tyre_age_laps < tyre_data[i - 1].tyre_age_laps or
                tyre_data[i].visual_tyre_compound != tyre_data[i - 1].visual_tyre_compound):
                stints.append(current_stint)
                current_stint = [tyre_data[i]]
            else:
                current_stint.append(tyre_data[i])

        if current_stint:
            stints.append(current_stint)

        return stints

    def add_prediction(
        self,
        current_state: Dict,
        future_laps: int = 10,
        model_path: Optional[str] = None
    ):
        """
        Add ML prediction overlay

        Args:
            current_state: Current tyre state
            future_laps: Number of laps to forecast
            model_path: Optional path to trained model
        """
        try:
            # Load model if not loaded
            if self.tyre_model is None:
                if model_path:
                    from ..ml.models import TyreWearModel
                    self.tyre_model = TyreWearModel(model_path=model_path)
                else:
                    # Try default path
                    try:
                        from ..ml.models import TyreWearModel
                        self.tyre_model = TyreWearModel(model_path='models/tyre_wear_model.pkl')
                    except:
                        print("Tyre wear model not available")
                        return

            # Get prediction
            forecast = self.tyre_model.predict_wear(current_state, future_laps=future_laps)

            # Extract current position
            current_lap = current_state.get('lap_number', 20)
            current_wear = current_state.get('avg_wear', 50.0)

            # Build forecast points
            forecast_laps = list(range(current_lap, current_lap + future_laps + 1))
            forecast_wear = [current_wear]

            wear_rate = forecast.get('wear_rate_per_lap', 2.0)
            for i in range(1, future_laps + 1):
                forecast_wear.append(current_wear + (wear_rate * i))

            # Plot prediction
            self.ax.plot(
                forecast_laps,
                forecast_wear,
                color='purple',
                linestyle='--',
                linewidth=2,
                marker='s',
                markersize=3,
                label=f'ML Prediction (+{wear_rate:.1f}%/lap)',
                alpha=0.7
            )

            # Confidence band (±5%)
            lower_band = [w - 5 for w in forecast_wear]
            upper_band = [w + 5 for w in forecast_wear]

            self.ax.fill_between(
                forecast_laps,
                lower_band,
                upper_band,
                color='purple',
                alpha=0.2,
                label='Confidence (±5%)'
            )

            # Update legend
            self.ax.legend(loc='upper left', fontsize=9)

            self.canvas.draw()

        except Exception as e:
            print(f"Error adding prediction: {e}")

    def _add_prediction(self):
        """Add prediction button handler"""
        if not self.session_id:
            return

        # Get latest tyre state from database
        try:
            repo = TyreDataRepository()
            tyre_data = repo.get_by_session_and_driver(self.session_id, self.driver_index)

            if not tyre_data:
                return

            # Latest data point
            latest = tyre_data[-1]

            current_state = {
                'lap_number': latest.tyre_age_laps,
                'tyre_age': latest.tyre_age_laps,
                'avg_wear': (latest.wear_fl + latest.wear_fr + latest.wear_rl + latest.wear_rr) / 4,
                'fuel_remaining': 50.0,  # Default
                'compound': latest.visual_tyre_compound
            }

            self.add_prediction(current_state, future_laps=15)

        except Exception as e:
            print(f"Error: {e}")

    def _refresh(self):
        """Refresh chart"""
        if self.session_id:
            self.plot_wear_history(self.session_id, self.driver_index)

    def _on_driver_changed(self, index):
        """Driver selection changed"""
        self.driver_index = index
        self._refresh()

    def _plot_demo_data(self):
        """Plot demo data when database unavailable"""
        self.ax.clear()

        # Demo wear curve
        laps = list(range(0, 30))
        wear_soft = [0 + (3.5 * lap) for lap in laps]
        wear_medium = [0 + (2.2 * lap) for lap in laps]

        self.ax.plot(laps[:15], wear_soft[:15], color='#FF0000', linewidth=2, marker='o', label='SOFT')
        self.ax.plot(laps[14:], wear_medium[14:], color='#FFFF00', linewidth=2, marker='o', label='MEDIUM')

        self.ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% Wear')
        self.ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Wear')

        self.ax.set_xlabel('Tyre Age (Laps)', fontsize=12)
        self.ax.set_ylabel('Tyre Wear (%)', fontsize=12)
        self.ax.set_title('Tyre Wear Evolution (Demo Data)', fontsize=14)
        self.ax.set_xlim(0, 35)
        self.ax.set_ylim(0, 120)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        self.canvas.draw()

    def _plot_no_data(self):
        """Plot when no data available"""
        self.ax.clear()
        self.ax.text(
            0.5, 0.5,
            'No tyre data available for this session',
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
