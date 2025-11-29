"""
Prediction Overlay Component

Real-time ML predictions overlaid on live telemetry data.

Features:
- Live updating of predictions
- Multiple prediction types (tyre, lap time, pit, outcome)
- Confidence indicators
- Historical accuracy tracking

Usage:
    from src.visualization import PredictionOverlay

    overlay = PredictionOverlay()
    overlay.set_predictor(predictor)
    overlay.update(current_state)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import QTimer
from typing import Dict, List, Optional

try:
    from ..ml.inference import RealTimePredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class PredictionOverlay(QWidget):
    """
    Real-time prediction overlay

    Shows live ML predictions with confidence indicators.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Matplotlib figure (2x2 subplots)
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)

        # Subplots
        self.ax_tyre = self.figure.add_subplot(2, 2, 1)
        self.ax_lap_time = self.figure.add_subplot(2, 2, 2)
        self.ax_pit = self.figure.add_subplot(2, 2, 3)
        self.ax_outcome = self.figure.add_subplot(2, 2, 4)

        # Predictor
        self.predictor: Optional[RealTimePredictor] = None

        # Data history
        self.history = {
            'laps': [],
            'tyre_wear': [],
            'tyre_pred': [],
            'lap_times': [],
            'lap_time_pred': [],
            'pit_lap': [],
            'win_prob': [],
            'podium_prob': [],
            'points_prob': []
        }

        # Setup UI
        self._setup_ui()

        # Auto-refresh timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._auto_refresh)
        self.auto_refresh_enabled = False

    def _setup_ui(self):
        """Setup widget layout"""
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Predictions: Not connected")
        layout.addWidget(self.status_label)

        # Canvas
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        # Initial empty plots
        self._init_plots()

    def _init_plots(self):
        """Initialize empty plots"""
        # Tyre wear prediction
        self.ax_tyre.set_title('Tyre Wear Forecast', fontweight='bold')
        self.ax_tyre.set_xlabel('Lap')
        self.ax_tyre.set_ylabel('Wear (%)')
        self.ax_tyre.grid(True, alpha=0.3)

        # Lap time forecast
        self.ax_lap_time.set_title('Lap Time Forecast', fontweight='bold')
        self.ax_lap_time.set_xlabel('Lap')
        self.ax_lap_time.set_ylabel('Lap Time (s)')
        self.ax_lap_time.grid(True, alpha=0.3)

        # Pit stop recommendation
        self.ax_pit.set_title('Pit Stop Strategy', fontweight='bold')
        self.ax_pit.set_xlabel('Pit Lap')
        self.ax_pit.set_ylabel('Expected Position')
        self.ax_pit.grid(True, alpha=0.3)

        # Race outcome probabilities
        self.ax_outcome.set_title('Race Outcome Probabilities', fontweight='bold')
        self.ax_outcome.set_xlabel('Lap')
        self.ax_outcome.set_ylabel('Probability (%)')
        self.ax_outcome.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def set_predictor(self, predictor: 'RealTimePredictor'):
        """
        Set the ML predictor

        Args:
            predictor: Loaded RealTimePredictor instance
        """
        self.predictor = predictor
        self.status_label.setText("Predictions: Connected")

    def update(self, current_state: Dict):
        """
        Update predictions with current state

        Args:
            current_state: Current race state
        """
        if not self.predictor:
            return

        try:
            # Get predictions
            predictions = self.predictor.predict(current_state)

            current_lap = current_state.get('current_lap', 0)

            # Update history
            self.history['laps'].append(current_lap)

            # Tyre wear
            if predictions.get('tyre_wear'):
                tw = predictions['tyre_wear']
                self.history['tyre_wear'].append(current_state.get('avg_wear', 50))
                self.history['tyre_pred'].append(tw.get('next_lap_wear', 50))

            # Lap time
            if predictions.get('lap_time'):
                lt = predictions['lap_time']
                self.history['lap_times'].append(current_state.get('current_lap_time', 85000) / 1000)
                self.history['lap_time_pred'].append(lt.get('next_lap_time_ms', 85000) / 1000)

            # Pit stop
            if predictions.get('pit_stop'):
                ps = predictions['pit_stop']
                self.history['pit_lap'].append(ps.get('optimal_pit_lap', current_lap + 10))

            # Race outcome
            if predictions.get('race_outcome'):
                ro = predictions['race_outcome']
                self.history['win_prob'].append(ro.get('win_probability', 0) * 100)
                self.history['podium_prob'].append(ro.get('podium_probability', 0) * 100)
                self.history['points_prob'].append(ro.get('points_probability', 0) * 100)

            # Update plots
            self._update_plots()

            self.status_label.setText(f"Predictions: Lap {current_lap} updated")

        except Exception as e:
            print(f"Error updating predictions: {e}")
            self.status_label.setText(f"Predictions: Error - {str(e)[:50]}")

    def _update_plots(self):
        """Update all plots with latest data"""
        # Tyre wear
        self.ax_tyre.clear()
        if self.history['tyre_wear']:
            self.ax_tyre.plot(
                self.history['laps'],
                self.history['tyre_wear'],
                'b-o',
                label='Actual',
                linewidth=2
            )
            self.ax_tyre.plot(
                self.history['laps'],
                self.history['tyre_pred'],
                'r--s',
                label='Predicted',
                linewidth=2,
                alpha=0.7
            )
            self.ax_tyre.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% Wear')
            self.ax_tyre.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Wear')
            self.ax_tyre.legend(loc='upper left')

        self.ax_tyre.set_title('Tyre Wear Forecast', fontweight='bold')
        self.ax_tyre.set_xlabel('Lap')
        self.ax_tyre.set_ylabel('Wear (%)')
        self.ax_tyre.grid(True, alpha=0.3)

        # Lap time
        self.ax_lap_time.clear()
        if self.history['lap_times']:
            self.ax_lap_time.plot(
                self.history['laps'],
                self.history['lap_times'],
                'b-o',
                label='Actual',
                linewidth=2
            )
            self.ax_lap_time.plot(
                self.history['laps'],
                self.history['lap_time_pred'],
                'r--s',
                label='Predicted',
                linewidth=2,
                alpha=0.7
            )
            self.ax_lap_time.legend(loc='upper left')

        self.ax_lap_time.set_title('Lap Time Forecast', fontweight='bold')
        self.ax_lap_time.set_xlabel('Lap')
        self.ax_lap_time.set_ylabel('Lap Time (s)')
        self.ax_lap_time.grid(True, alpha=0.3)

        # Pit stop
        self.ax_pit.clear()
        if self.history['pit_lap']:
            # Show optimal pit lap evolution
            self.ax_pit.plot(
                self.history['laps'],
                self.history['pit_lap'],
                'g-o',
                label='Optimal Pit Lap',
                linewidth=2
            )
            self.ax_pit.legend()

        self.ax_pit.set_title('Pit Stop Strategy', fontweight='bold')
        self.ax_pit.set_xlabel('Current Lap')
        self.ax_pit.set_ylabel('Recommended Pit Lap')
        self.ax_pit.grid(True, alpha=0.3)

        # Race outcome
        self.ax_outcome.clear()
        if self.history['win_prob']:
            self.ax_outcome.plot(
                self.history['laps'],
                self.history['win_prob'],
                'gold',
                label='Win',
                linewidth=2,
                marker='o'
            )
            self.ax_outcome.plot(
                self.history['laps'],
                self.history['podium_prob'],
                'silver',
                label='Podium',
                linewidth=2,
                marker='s'
            )
            self.ax_outcome.plot(
                self.history['laps'],
                self.history['points_prob'],
                'brown',
                label='Points',
                linewidth=2,
                marker='^'
            )
            self.ax_outcome.legend(loc='best')

        self.ax_outcome.set_title('Race Outcome Probabilities', fontweight='bold')
        self.ax_outcome.set_xlabel('Lap')
        self.ax_outcome.set_ylabel('Probability (%)')
        self.ax_outcome.set_ylim(0, 105)
        self.ax_outcome.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def start_auto_refresh(self, interval_ms: int = 5000):
        """
        Start auto-refresh timer

        Args:
            interval_ms: Refresh interval in milliseconds
        """
        self.timer.start(interval_ms)
        self.auto_refresh_enabled = True

    def stop_auto_refresh(self):
        """Stop auto-refresh timer"""
        self.timer.stop()
        self.auto_refresh_enabled = False

    def _auto_refresh(self):
        """Auto-refresh callback"""
        # In real application, this would fetch latest state from live data
        pass

    def clear_history(self):
        """Clear prediction history"""
        for key in self.history:
            self.history[key] = []

        self._init_plots()
        self.status_label.setText("Predictions: History cleared")
