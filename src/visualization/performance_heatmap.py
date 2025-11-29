"""
Performance Heatmap Component

Sector-by-sector performance analysis with heatmap visualization.

Features:
- Sector time heatmap across laps
- Best theoretical lap builder
- Driver comparison heatmap
- Identify strong/weak sectors
- Consistency analysis

Usage:
    from src.visualization import PerformanceHeatmap

    heatmap = PerformanceHeatmap()
    heatmap.plot_sector_performance(session_id=1, driver_index=0)
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
    from ..analysis import LapAnalytics
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class PerformanceHeatmap(QWidget):
    """Sector performance heatmap visualization"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Data
        self.session_id = None
        self.driver_index = 0

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

        # View mode
        self.view_label = QLabel("View:")
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Sector Times", "Consistency", "Best Lap"])
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)

        controls_layout.addWidget(self.driver_label)
        controls_layout.addWidget(self.driver_combo)
        controls_layout.addWidget(self.view_label)
        controls_layout.addWidget(self.view_combo)
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addStretch()

        # Canvas
        layout.addLayout(controls_layout)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def plot_sector_performance(self, session_id: int, driver_index: int = 0):
        """
        Plot sector performance heatmap

        Args:
            session_id: Database session ID
            driver_index: Driver index (0-21)
        """
        if not DATABASE_AVAILABLE:
            self._plot_demo()
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

            # Extract sector times
            laps = []
            sector1_times = []
            sector2_times = []
            sector3_times = []

            for lap in lap_data:
                if lap.sector1_time_ms > 0 and lap.sector2_time_ms > 0 and lap.sector3_time_ms > 0:
                    laps.append(lap.current_lap_num)
                    sector1_times.append(lap.sector1_time_ms / 1000)  # Convert to seconds
                    sector2_times.append(lap.sector2_time_ms / 1000)
                    sector3_times.append(lap.sector3_time_ms / 1000)

            if not laps:
                self._plot_no_data()
                return

            # Create heatmap data
            heatmap_data = np.array([sector1_times, sector2_times, sector3_times])

            # Normalize by column (each lap)
            # Color based on deviation from mean
            normalized_data = np.zeros_like(heatmap_data)
            for i in range(heatmap_data.shape[0]):
                row = heatmap_data[i]
                mean = np.mean(row)
                std = np.std(row)
                if std > 0:
                    normalized_data[i] = (row - mean) / std
                else:
                    normalized_data[i] = row - mean

            # Plot heatmap
            self.ax.clear()

            im = self.ax.imshow(
                normalized_data,
                cmap='RdYlGn_r',  # Red = slow, Green = fast
                aspect='auto',
                interpolation='nearest'
            )

            # Set ticks
            self.ax.set_yticks([0, 1, 2])
            self.ax.set_yticklabels(['Sector 1', 'Sector 2', 'Sector 3'])
            self.ax.set_xticks(range(0, len(laps), max(1, len(laps) // 10)))
            self.ax.set_xticklabels([laps[i] for i in range(0, len(laps), max(1, len(laps) // 10))])

            # Labels
            self.ax.set_xlabel('Lap Number', fontsize=12)
            self.ax.set_title(
                f'Sector Performance Heatmap - Driver {driver_index}\n(Green = Fast, Red = Slow)',
                fontsize=14,
                fontweight='bold'
            )

            # Colorbar
            cbar = self.figure.colorbar(im, ax=self.ax)
            cbar.set_label('Performance (σ from mean)', rotation=270, labelpad=20)

            # Add best sector times annotation
            best_s1 = min(sector1_times)
            best_s2 = min(sector2_times)
            best_s3 = min(sector3_times)
            ideal_lap = best_s1 + best_s2 + best_s3

            textstr = f'Best Sectors:\nS1: {best_s1:.3f}s\nS2: {best_s2:.3f}s\nS3: {best_s3:.3f}s\nIdeal: {ideal_lap:.3f}s'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            self.ax.text(
                1.15, 0.5, textstr,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment='center',
                bbox=props
            )

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error plotting heatmap: {e}")
            self._plot_error(str(e))

    def _refresh(self):
        """Refresh chart"""
        if self.session_id:
            self.plot_sector_performance(self.session_id, self.driver_index)

    def _on_driver_changed(self, index):
        """Driver selection changed"""
        self.driver_index = index
        self._refresh()

    def _on_view_changed(self, index):
        """View mode changed"""
        self._refresh()

    def _plot_demo(self):
        """Plot demo heatmap"""
        self.ax.clear()

        # Generate demo sector times (30 laps, 3 sectors)
        np.random.seed(42)
        laps = 30
        sectors = 3

        # Base times with some variation
        base_times = [25, 30, 28]  # Base time for each sector
        heatmap_data = []

        for sector_idx in range(sectors):
            base = base_times[sector_idx]
            times = [base + np.random.normal(0, 0.5) for _ in range(laps)]
            heatmap_data.append(times)

        heatmap_data = np.array(heatmap_data)

        # Normalize
        normalized_data = np.zeros_like(heatmap_data)
        for i in range(sectors):
            row = heatmap_data[i]
            mean = np.mean(row)
            std = np.std(row)
            normalized_data[i] = (row - mean) / std if std > 0 else row - mean

        # Plot
        im = self.ax.imshow(
            normalized_data,
            cmap='RdYlGn_r',
            aspect='auto',
            interpolation='nearest'
        )

        self.ax.set_yticks([0, 1, 2])
        self.ax.set_yticklabels(['Sector 1', 'Sector 2', 'Sector 3'])
        self.ax.set_xlabel('Lap Number', fontsize=12)
        self.ax.set_title('Sector Performance Heatmap (Demo Data)', fontsize=14, fontweight='bold')

        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.set_label('Performance', rotation=270, labelpad=20)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_no_data(self):
        """Plot when no data available"""
        self.ax.clear()
        self.ax.text(
            0.5, 0.5,
            'No sector data available for this session',
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
