"""
Strategy Simulator Component

Visual comparison of pit strategies.

Features:
- Side-by-side strategy comparison
- Race time simulation
- Position evolution
- Risk analysis visualization
- Interactive "what-if" scenarios

Usage:
    from src.visualization import StrategySimulator

    sim = StrategySimulator()
    sim.compare_strategies(current_state, strategies)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QComboBox, QGroupBox
)
from typing import Dict, List, Optional

try:
    from ..ml.models import StrategyRecommender
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class StrategySimulator(QWidget):
    """Strategy comparison visualization"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Figure with 2 subplots
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax_timeline = self.figure.add_subplot(2, 1, 1)
        self.ax_comparison = self.figure.add_subplot(2, 1, 2)

        # Strategy recommender
        self.recommender: Optional[StrategyRecommender] = None

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup widget layout"""
        layout = QVBoxLayout()

        # Controls
        controls_group = QGroupBox("Scenario Parameters")
        controls_layout = QHBoxLayout()

        # Current lap
        controls_layout.addWidget(QLabel("Current Lap:"))
        self.current_lap_spin = QSpinBox()
        self.current_lap_spin.setRange(1, 70)
        self.current_lap_spin.setValue(15)
        controls_layout.addWidget(self.current_lap_spin)

        # Total laps
        controls_layout.addWidget(QLabel("Total Laps:"))
        self.total_laps_spin = QSpinBox()
        self.total_laps_spin.setRange(20, 100)
        self.total_laps_spin.setValue(50)
        controls_layout.addWidget(self.total_laps_spin)

        # Current position
        controls_layout.addWidget(QLabel("Position:"))
        self.position_spin = QSpinBox()
        self.position_spin.setRange(1, 20)
        self.position_spin.setValue(5)
        controls_layout.addWidget(self.position_spin)

        # Simulate button
        self.simulate_btn = QPushButton("Simulate Strategies")
        self.simulate_btn.clicked.connect(self._simulate)
        controls_layout.addWidget(self.simulate_btn)

        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)

        # Canvas
        layout.addWidget(controls_group)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        # Initial plot
        self._init_plots()

    def _init_plots(self):
        """Initialize empty plots"""
        self.ax_timeline.set_title('Strategy Timeline', fontweight='bold')
        self.ax_timeline.set_xlabel('Lap')
        self.ax_timeline.set_ylabel('Position')
        self.ax_timeline.grid(True, alpha=0.3)

        self.ax_comparison.set_title('Strategy Comparison', fontweight='bold')
        self.ax_comparison.set_ylabel('Strategy')
        self.ax_comparison.set_xlabel('Expected Position')
        self.ax_comparison.grid(True, alpha=0.3, axis='x')

        self.figure.tight_layout()
        self.canvas.draw()

    def compare_strategies(self, current_state: Dict, max_strategies: int = 5):
        """Compare multiple strategies"""
        if not ML_AVAILABLE:
            self._plot_demo()
            return

        try:
            if self.recommender is None:
                self.recommender = StrategyRecommender()

            # Get strategy recommendations
            result = self.recommender.recommend_strategy(current_state, max_strategies=max_strategies)

            recommended = result.get('recommended_strategy')
            alternatives = result.get('alternative_strategies', [])

            all_strategies = [recommended] + alternatives if recommended else alternatives

            # Plot timeline
            self._plot_strategy_timeline(all_strategies, current_state)

            # Plot comparison
            self._plot_strategy_comparison(all_strategies)

            self.canvas.draw()

        except Exception as e:
            print(f"Error comparing strategies: {e}")

    def _plot_strategy_timeline(self, strategies, current_state):
        """Plot position evolution for each strategy"""
        self.ax_timeline.clear()

        current_lap = current_state.get('current_lap', 15)
        total_laps = current_state.get('total_laps', 50)
        current_position = current_state.get('current_position', 5)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, strategy in enumerate(strategies):
            # Simulate position evolution
            laps = list(range(current_lap, total_laps + 1))
            positions = [current_position] * len(laps)

            # Adjust at pit stops
            for pit_lap in strategy.pit_laps:
                if pit_lap in laps:
                    pit_idx = laps.index(pit_lap)
                    # Position drop during pit
                    positions[pit_idx] = min(20, current_position + 3)
                    # Recovery
                    for i in range(pit_idx + 1, len(positions)):
                        positions[i] = max(1, positions[pit_idx] - (i - pit_idx) * 0.2)

            # Final position
            if positions:
                positions[-1] = strategy.expected_position

            # Plot
            color = colors[idx % len(colors)]
            label = f"{strategy.name} → P{strategy.expected_position}"

            self.ax_timeline.plot(
                laps,
                positions,
                color=color,
                linewidth=2,
                marker='o',
                markersize=3,
                label=label
            )

            # Mark pit stops
            for pit_lap in strategy.pit_laps:
                if pit_lap in laps:
                    self.ax_timeline.axvline(x=pit_lap, color=color, linestyle='--', alpha=0.3)

        self.ax_timeline.set_title('Strategy Timeline', fontweight='bold')
        self.ax_timeline.set_xlabel('Lap')
        self.ax_timeline.set_ylabel('Position')
        self.ax_timeline.invert_yaxis()  # Lower position = better
        self.ax_timeline.set_ylim(20, 0)
        self.ax_timeline.grid(True, alpha=0.3)
        self.ax_timeline.legend(loc='best', fontsize=8)

    def _plot_strategy_comparison(self, strategies):
        """Plot horizontal bar comparison"""
        self.ax_comparison.clear()

        if not strategies:
            return

        # Extract data
        names = [s.name[:30] for s in strategies]  # Truncate long names
        positions = [s.expected_position for s in strategies]
        podium_probs = [s.podium_probability * 100 for s in strategies]

        # Color by risk
        colors = []
        for s in strategies:
            if s.risk_level == 'LOW':
                colors.append('#2ca02c')  # Green
            elif s.risk_level == 'MEDIUM':
                colors.append('#ff7f0e')  # Orange
            else:
                colors.append('#d62728')  # Red

        y_pos = np.arange(len(names))

        # Plot bars (expected position)
        bars = self.ax_comparison.barh(y_pos, positions, color=colors, alpha=0.7)

        # Add podium probability as text
        for i, (pos, prob) in enumerate(zip(positions, podium_probs)):
            self.ax_comparison.text(
                pos + 0.3, i,
                f'P{pos:.0f} ({prob:.0f}%)',
                va='center',
                fontsize=9
            )

        self.ax_comparison.set_yticks(y_pos)
        self.ax_comparison.set_yticklabels(names, fontsize=9)
        self.ax_comparison.set_xlabel('Expected Position', fontsize=10)
        self.ax_comparison.set_title('Strategy Comparison', fontweight='bold')
        self.ax_comparison.invert_xaxis()  # Lower position = better
        self.ax_comparison.grid(True, alpha=0.3, axis='x')

        # Add legend for risk
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', label='Low Risk'),
            Patch(facecolor='#ff7f0e', label='Medium Risk'),
            Patch(facecolor='#d62728', label='High Risk')
        ]
        self.ax_comparison.legend(handles=legend_elements, loc='lower right', fontsize=8)

        self.figure.tight_layout()

    def _simulate(self):
        """Simulate button clicked"""
        current_state = {
            'current_lap': self.current_lap_spin.value(),
            'total_laps': self.total_laps_spin.value(),
            'current_position': self.position_spin.value(),
            'current_compound': 'MEDIUM',
            'tyre_age': 10,
            'base_lap_time': 87000
        }

        self.compare_strategies(current_state, max_strategies=5)

    def _plot_demo(self):
        """Plot demo data"""
        self.ax_timeline.clear()
        self.ax_comparison.clear()

        # Demo timeline
        laps = list(range(15, 51))
        pos_1_stop = [5] * len(laps)
        pos_2_stop = [5] * len(laps)

        # 1-stop at lap 25
        for i, lap in enumerate(laps):
            if lap >= 25:
                pos_1_stop[i] = 4  # Better final position

        # 2-stop at laps 20, 35
        for i, lap in enumerate(laps):
            if 20 <= lap < 22:
                pos_2_stop[i] = 8  # Drop during pit
            elif 35 <= lap < 37:
                pos_2_stop[i] = 7
            elif lap >= 37:
                pos_2_stop[i] = 5  # Back to starting position

        self.ax_timeline.plot(laps, pos_1_stop, 'b-o', label='1-stop → P4', linewidth=2)
        self.ax_timeline.plot(laps, pos_2_stop, 'r-s', label='2-stop → P5', linewidth=2)
        self.ax_timeline.set_title('Strategy Timeline (Demo)', fontweight='bold')
        self.ax_timeline.set_xlabel('Lap')
        self.ax_timeline.set_ylabel('Position')
        self.ax_timeline.invert_yaxis()
        self.ax_timeline.set_ylim(10, 1)
        self.ax_timeline.grid(True, alpha=0.3)
        self.ax_timeline.legend()

        # Demo comparison
        strategies = ['1-stop M-H (Early)', '1-stop M-H (Late)', '2-stop S-M-S']
        positions = [4, 5, 5]
        colors = ['#2ca02c', '#ff7f0e', '#d62728']

        self.ax_comparison.barh(strategies, positions, color=colors, alpha=0.7)
        self.ax_comparison.set_xlabel('Expected Position')
        self.ax_comparison.set_title('Strategy Comparison (Demo)', fontweight='bold')
        self.ax_comparison.invert_xaxis()
        self.ax_comparison.grid(True, alpha=0.3, axis='x')

        self.figure.tight_layout()
        self.canvas.draw()
