"""
Strategy Window

Interactive strategy comparison and "what-if" analysis.

Features:
- Strategy simulator with visual comparison
- What-if scenario testing
- Pit timing optimization
- Real-time strategy recommendations
- Risk assessment

Usage:
    from src.windows import StrategyWindow

    window = StrategyWindow()
    window.set_current_state(current_state)
    window.show()
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QComboBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt
from typing import Dict, Optional

try:
    from ..visualization import StrategySimulator
    from ..ml.models import StrategyRecommender, PitStopOptimizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class StrategyWindow(QWidget):
    """
    Strategy comparison and optimization dashboard

    Interactive strategy analysis with ML recommendations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.recommender: Optional['StrategyRecommender'] = None
        self.pit_optimizer: Optional['PitStopOptimizer'] = None
        self.current_state: Dict = {}

        # Setup UI
        self._setup_ui()

        # Initialize ML models
        if ML_AVAILABLE:
            try:
                self.recommender = StrategyRecommender()
                self.pit_optimizer = PitStopOptimizer()
            except:
                pass

    def _setup_ui(self):
        """Setup window layout"""
        main_layout = QHBoxLayout()

        # Left side: Strategy simulator
        left_side = QVBoxLayout()

        if ML_AVAILABLE:
            self.strategy_simulator = StrategySimulator()
            left_side.addWidget(self.strategy_simulator, stretch=3)
        else:
            placeholder = QLabel("ML components not available.\nInstall required dependencies.")
            placeholder.setAlignment(Qt.AlignCenter)
            left_side.addWidget(placeholder, stretch=3)

        # Right side: Controls and recommendations
        right_side = QVBoxLayout()

        # Current state
        state_group = self._create_state_controls()
        right_side.addWidget(state_group)

        # Strategy recommendations table
        rec_group = self._create_recommendations_table()
        right_side.addWidget(rec_group, stretch=2)

        # Pit stop analysis
        pit_group = self._create_pit_analysis()
        right_side.addWidget(pit_group, stretch=1)

        # Add sides to main layout
        left_widget = QWidget()
        left_widget.setLayout(left_side)
        right_widget = QWidget()
        right_widget.setLayout(right_side)

        main_layout.addWidget(left_widget, stretch=2)
        main_layout.addWidget(right_widget, stretch=1)

        self.setLayout(main_layout)
        self.setWindowTitle("F1 Strategy Comparison")

    def _create_state_controls(self):
        """Create current state controls"""
        group = QGroupBox("Current State")
        layout = QVBoxLayout()

        # Grid layout for parameters
        params_layout = QHBoxLayout()

        # Lap
        params_layout.addWidget(QLabel("Lap:"))
        self.lap_spin = QSpinBox()
        self.lap_spin.setRange(1, 70)
        self.lap_spin.setValue(15)
        params_layout.addWidget(self.lap_spin)

        # Position
        params_layout.addWidget(QLabel("Pos:"))
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(1, 20)
        self.pos_spin.setValue(5)
        params_layout.addWidget(self.pos_spin)

        layout.addLayout(params_layout)

        # Total laps and compound
        params_layout2 = QHBoxLayout()

        params_layout2.addWidget(QLabel("Total Laps:"))
        self.total_laps_spin = QSpinBox()
        self.total_laps_spin.setRange(20, 100)
        self.total_laps_spin.setValue(50)
        params_layout2.addWidget(self.total_laps_spin)

        params_layout2.addWidget(QLabel("Compound:"))
        self.compound_combo = QComboBox()
        self.compound_combo.addItems(['SOFT', 'MEDIUM', 'HARD'])
        self.compound_combo.setCurrentText('MEDIUM')
        params_layout2.addWidget(self.compound_combo)

        layout.addLayout(params_layout2)

        # Analyze button
        self.analyze_btn = QPushButton("Analyze Strategies")
        self.analyze_btn.clicked.connect(self._analyze_strategies)
        layout.addWidget(self.analyze_btn)

        group.setLayout(layout)
        return group

    def _create_recommendations_table(self):
        """Create strategy recommendations table"""
        group = QGroupBox("Strategy Recommendations")
        layout = QVBoxLayout()

        self.rec_table = QTableWidget()
        self.rec_table.setColumnCount(5)
        self.rec_table.setHorizontalHeaderLabels([
            'Strategy', 'Pit Laps', 'Expected Pos', 'Podium %', 'Risk'
        ])
        self.rec_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rec_table.setAlternatingRowColors(True)

        layout.addWidget(self.rec_table)

        group.setLayout(layout)
        return group

    def _create_pit_analysis(self):
        """Create pit stop analysis section"""
        group = QGroupBox("Pit Stop Analysis")
        layout = QVBoxLayout()

        self.pit_text = QTextEdit()
        self.pit_text.setReadOnly(True)
        self.pit_text.setMaximumHeight(200)

        layout.addWidget(self.pit_text)

        group.setLayout(layout)
        return group

    def set_current_state(self, state: Dict):
        """
        Set current race state

        Args:
            state: Current race state dictionary
        """
        self.current_state = state

        # Update UI
        if 'current_lap' in state:
            self.lap_spin.setValue(state['current_lap'])
        if 'total_laps' in state:
            self.total_laps_spin.setValue(state['total_laps'])
        if 'current_position' in state:
            self.pos_spin.setValue(state['current_position'])
        if 'current_compound' in state:
            idx = self.compound_combo.findText(state['current_compound'])
            if idx >= 0:
                self.compound_combo.setCurrentIndex(idx)

    def _analyze_strategies(self):
        """Analyze button clicked"""
        if not ML_AVAILABLE:
            return

        # Build state from UI
        state = {
            'current_lap': self.lap_spin.value(),
            'total_laps': self.total_laps_spin.value(),
            'current_position': self.pos_spin.value(),
            'current_compound': self.compound_combo.currentText(),
            'tyre_age': 10,  # Default
            'base_lap_time': 87000  # Default
        }

        self.current_state = state

        # Get strategy recommendations
        if self.recommender:
            try:
                result = self.recommender.recommend_strategy(state, max_strategies=5)

                # Update simulator
                if hasattr(self, 'strategy_simulator'):
                    self.strategy_simulator.compare_strategies(state, max_strategies=5)

                # Update recommendations table
                self._update_recommendations_table(result)

            except Exception as e:
                print(f"Error analyzing strategies: {e}")

        # Get pit stop analysis
        if self.pit_optimizer:
            try:
                pit_result = self.pit_optimizer.optimize_pit_window(state)
                self._update_pit_analysis(pit_result)
            except Exception as e:
                print(f"Error analyzing pit stops: {e}")

    def _update_recommendations_table(self, result: Dict):
        """Update recommendations table with results"""
        recommended = result.get('recommended_strategy')
        alternatives = result.get('alternative_strategies', [])

        all_strategies = []
        if recommended:
            all_strategies.append(recommended)
        all_strategies.extend(alternatives)

        # Clear and populate table
        self.rec_table.setRowCount(len(all_strategies))

        for row, strategy in enumerate(all_strategies):
            # Strategy name
            self.rec_table.setItem(row, 0, QTableWidgetItem(strategy.name[:30]))

            # Pit laps
            pit_laps_str = ', '.join(map(str, strategy.pit_laps)) if strategy.pit_laps else 'None'
            self.rec_table.setItem(row, 1, QTableWidgetItem(pit_laps_str))

            # Expected position
            self.rec_table.setItem(row, 2, QTableWidgetItem(f"P{strategy.expected_position}"))

            # Podium probability
            podium_pct = f"{strategy.podium_probability * 100:.0f}%"
            self.rec_table.setItem(row, 3, QTableWidgetItem(podium_pct))

            # Risk level
            risk_item = QTableWidgetItem(strategy.risk_level)
            if strategy.risk_level == 'HIGH':
                risk_item.setForeground(Qt.red)
            elif strategy.risk_level == 'MEDIUM':
                risk_item.setForeground(Qt.darkYellow)
            else:
                risk_item.setForeground(Qt.darkGreen)
            self.rec_table.setItem(row, 4, risk_item)

            # Highlight recommended (first row)
            if row == 0:
                for col in range(5):
                    item = self.rec_table.item(row, col)
                    if item:
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)

    def _update_pit_analysis(self, pit_result: Dict):
        """Update pit stop analysis text"""
        text = "🏁 PIT STOP OPTIMIZATION\n\n"

        text += f"Optimal Pit Lap: {pit_result['optimal_pit_lap']}\n"
        text += f"Expected Position: P{pit_result['expected_position']:.1f}\n"
        text += f"Position Change: {pit_result['position_change']:+.1f}\n\n"

        text += f"Undercut Potential: {pit_result['undercut_potential']:.2f}\n"
        text += f"Overcut Potential: {pit_result['overcut_potential']:.2f}\n"
        text += f"Risk Score: {pit_result['risk_score']:.2f}\n\n"

        text += f"Recommendation:\n{pit_result['recommendation']}\n\n"

        # Alternatives
        text += "Alternative Pit Laps:\n"
        for alt in pit_result.get('alternatives', [])[:3]:
            text += f"  Lap {alt['pit_lap']}: P{alt['expected_position']:.1f} (risk {alt['risk_score']:.2f})\n"

        self.pit_text.setText(text)
