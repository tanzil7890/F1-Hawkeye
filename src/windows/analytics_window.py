"""
Analytics Window

Historical telemetry analysis dashboard.

Features:
- Tyre wear analysis
- Pace evolution
- Sector performance heatmap
- Session selection
- Multi-driver comparison

Usage:
    from src.windows import AnalyticsWindow

    window = AnalyticsWindow()
    window.load_session(session_id=1)
    window.show()
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QComboBox, QSpinBox, QGroupBox
)
from PySide6.QtCore import Qt

try:
    from ..visualization import TyreWearChart, PaceEvolutionChart, PerformanceHeatmap
    from ..database import db_manager
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class AnalyticsWindow(QWidget):
    """
    Historical analysis dashboard

    Combines multiple visualization components for telemetry analysis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_session_id = None

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup window layout"""
        layout = QVBoxLayout()

        # Header with session selection
        header = self._create_header()
        layout.addWidget(header)

        # Tabs for different analyses
        self.tabs = QTabWidget()

        if VISUALIZATION_AVAILABLE:
            # Tyre Analysis Tab
            self.tyre_chart = TyreWearChart()
            self.tabs.addTab(self.tyre_chart, "Tyre Wear Analysis")

            # Pace Analysis Tab
            self.pace_chart = PaceEvolutionChart()
            self.tabs.addTab(self.pace_chart, "Pace Evolution")

            # Performance Heatmap Tab
            self.heatmap = PerformanceHeatmap()
            self.tabs.addTab(self.heatmap, "Sector Performance")
        else:
            # Placeholder if visualization not available
            placeholder = QLabel("Visualization components not available.\nInstall required dependencies.")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tabs.addTab(placeholder, "Analytics")

        layout.addWidget(self.tabs)

        self.setLayout(layout)
        self.setWindowTitle("F1 Telemetry Analytics")

    def _create_header(self):
        """Create header with controls"""
        header_group = QGroupBox("Session Selection")
        header_layout = QHBoxLayout()

        # Session selector
        header_layout.addWidget(QLabel("Session ID:"))
        self.session_spin = QSpinBox()
        self.session_spin.setRange(1, 10000)
        self.session_spin.setValue(1)
        header_layout.addWidget(self.session_spin)

        # Load button
        self.load_btn = QPushButton("Load Session")
        self.load_btn.clicked.connect(self._load_session)
        header_layout.addWidget(self.load_btn)

        # Recent sessions dropdown
        if VISUALIZATION_AVAILABLE:
            try:
                sessions = db_manager.get_recent_sessions(limit=10)
                if sessions:
                    header_layout.addWidget(QLabel("Recent:"))
                    self.recent_combo = QComboBox()
                    for s in sessions:
                        self.recent_combo.addItem(
                            f"Session {s.id}: {s.track_name} ({s.session_type})",
                            s.id
                        )
                    self.recent_combo.currentIndexChanged.connect(self._on_recent_selected)
                    header_layout.addWidget(self.recent_combo)
            except:
                pass  # Database not available

        header_layout.addStretch()

        # Info label
        self.info_label = QLabel("No session loaded")
        header_layout.addWidget(self.info_label)

        header_group.setLayout(header_layout)
        return header_group

    def load_session(self, session_id: int):
        """
        Load session data into all charts

        Args:
            session_id: Database session ID
        """
        if not VISUALIZATION_AVAILABLE:
            return

        try:
            self.current_session_id = session_id

            # Load into each chart
            self.tyre_chart.plot_wear_history(session_id)
            self.pace_chart.plot_lap_times(session_id)
            self.heatmap.plot_sector_performance(session_id)

            # Update info
            try:
                session = db_manager.get_session_by_id(session_id)
                if session:
                    self.info_label.setText(
                        f"Loaded: {session.track_name} - {session.session_type}"
                    )
                else:
                    self.info_label.setText(f"Loaded: Session {session_id}")
            except:
                self.info_label.setText(f"Loaded: Session {session_id}")

        except Exception as e:
            print(f"Error loading session: {e}")
            self.info_label.setText(f"Error: {str(e)[:50]}")

    def _load_session(self):
        """Load button clicked"""
        session_id = self.session_spin.value()
        self.load_session(session_id)

    def _on_recent_selected(self, index):
        """Recent session selected"""
        if hasattr(self, 'recent_combo'):
            session_id = self.recent_combo.itemData(index)
            if session_id:
                self.session_spin.setValue(session_id)
                self.load_session(session_id)
