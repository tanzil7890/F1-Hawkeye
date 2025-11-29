"""
Prediction Window

ML predictions dashboard with real-time updates.

Features:
- Live ML predictions (tyre, lap time, pit, outcome)
- Prediction history tracking
- Model loading/status
- Confidence indicators
- Auto-refresh capability

Usage:
    from src.windows import PredictionWindow

    window = PredictionWindow()
    window.load_models('models/')
    window.update_predictions(current_state)
    window.show()
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QTextEdit, QFileDialog, QCheckBox
)
from PySide6.QtCore import Qt, QTimer
from typing import Dict, Optional

try:
    from ..visualization import PredictionOverlay
    from ..ml.inference import RealTimePredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class PredictionWindow(QWidget):
    """
    ML predictions dashboard

    Real-time predictions with visualization.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.predictor: Optional['RealTimePredictor'] = None

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup window layout"""
        main_layout = QVBoxLayout()

        # Model loading controls
        model_controls = self._create_model_controls()
        main_layout.addWidget(model_controls)

        # Prediction overlay (charts)
        if ML_AVAILABLE:
            self.prediction_overlay = PredictionOverlay()
            main_layout.addWidget(self.prediction_overlay, stretch=3)
        else:
            placeholder = QLabel("ML components not available.\nInstall required dependencies (scikit-learn, xgboost).")
            placeholder.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(placeholder, stretch=3)

        # Prediction log
        log_group = QGroupBox("Prediction Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group, stretch=1)

        self.setLayout(main_layout)
        self.setWindowTitle("F1 Telemetry ML Predictions")

    def _create_model_controls(self):
        """Create model loading controls"""
        controls_group = QGroupBox("Model Controls")
        layout = QHBoxLayout()

        # Load models button
        self.load_models_btn = QPushButton("Load Models...")
        self.load_models_btn.clicked.connect(self._load_models)
        layout.addWidget(self.load_models_btn)

        # Model status
        layout.addWidget(QLabel("Status:"))
        self.model_status_label = QLabel("Not Loaded")
        self.model_status_label.setStyleSheet("color: orange;")
        layout.addWidget(self.model_status_label)

        # Auto-refresh toggle
        if ML_AVAILABLE:
            self.auto_refresh_cb = QCheckBox("Auto-Refresh (5s)")
            self.auto_refresh_cb.stateChanged.connect(self._on_auto_refresh_changed)
            layout.addWidget(self.auto_refresh_cb)

        # Clear history
        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self._clear_history)
        layout.addWidget(self.clear_btn)

        # Update now button
        self.update_btn = QPushButton("Update Now")
        self.update_btn.clicked.connect(self._update_now)
        layout.addWidget(self.update_btn)

        layout.addStretch()

        controls_group.setLayout(layout)
        return controls_group

    def load_models(self, models_dir: str):
        """
        Load ML models from directory

        Args:
            models_dir: Path to models directory
        """
        if not ML_AVAILABLE:
            self._log("ML components not available")
            return

        try:
            self._log(f"Loading models from {models_dir}...")

            # Create predictor
            self.predictor = RealTimePredictor()
            self.predictor.load_models(models_dir)

            # Connect to overlay
            self.prediction_overlay.set_predictor(self.predictor)

            # Update status
            status = self.predictor.get_status()
            if status['models_loaded']:
                self.model_status_label.setText("Loaded ✓")
                self.model_status_label.setStyleSheet("color: green;")
                self._log("Models loaded successfully")
            else:
                self.model_status_label.setText("Partial")
                self.model_status_label.setStyleSheet("color: orange;")
                self._log("Some models failed to load")

        except Exception as e:
            self._log(f"Error loading models: {e}")
            self.model_status_label.setText("Error ✗")
            self.model_status_label.setStyleSheet("color: red;")

    def update_predictions(self, current_state: Dict):
        """
        Update predictions with current state

        Args:
            current_state: Current race state
        """
        if not self.predictor or not ML_AVAILABLE:
            self._log("Predictor not initialized")
            return

        try:
            # Update overlay
            self.prediction_overlay.update(current_state)

            # Log summary
            lap = current_state.get('current_lap', 0)
            pos = current_state.get('current_position', 0)
            self._log(f"Lap {lap}, P{pos}: Predictions updated")

        except Exception as e:
            self._log(f"Error updating predictions: {e}")

    def _load_models(self):
        """Load models button clicked"""
        models_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Models Directory",
            "models"
        )

        if models_dir:
            self.load_models(models_dir)

    def _update_now(self):
        """Update now button clicked"""
        # Demo update with simulated state
        demo_state = {
            'current_lap': 25,
            'current_position': 5,
            'tyre_age': 15,
            'avg_wear': 52.0,
            'total_laps': 50,
            'current_lap_time': 87000,
            'base_lap_time': 85000,
            'fuel_remaining': 35.0
        }

        self.update_predictions(demo_state)

    def _clear_history(self):
        """Clear prediction history"""
        if ML_AVAILABLE and hasattr(self, 'prediction_overlay'):
            self.prediction_overlay.clear_history()
            self._log("Prediction history cleared")

    def _on_auto_refresh_changed(self, state):
        """Auto-refresh toggle changed"""
        if not ML_AVAILABLE:
            return

        if state == Qt.Checked:
            self.prediction_overlay.start_auto_refresh(interval_ms=5000)
            self._log("Auto-refresh enabled (5s interval)")
        else:
            self.prediction_overlay.stop_auto_refresh()
            self._log("Auto-refresh disabled")

    def _log(self, message: str):
        """Add message to log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
