from PySide6.QtCore import QSize, QAbstractTableModel
from PySide6.QtGui import QFont, Qt, QAction
from PySide6.QtWidgets import (
    QMainWindow, QTableView, QVBoxLayout, QWidget, QTabWidget, QHBoxLayout, QLabel, QAbstractItemView,
    QTabBar, QStackedWidget, QSizePolicy, QMessageBox, QFileDialog, QInputDialog, QListWidget
)

from src.packet_processing.packet_management import *
from src.table_models.DamageTableModel import DamageTableModel
from src.table_models.ERSAndFuelTableMable import ERSAndFuelTableModel
from src.table_models.LapTableModel import LapTableModel

from src.table_models.MainTableModel import MainTableModel
from src.table_models.PacketReceptionTableModel import PacketReceptionTableModel
from src.table_models.TemperatureTableModel import TemperatureTableModel
from src.table_models.WeatherForecastTableModel import WeatherForecastTableModel
from src.table_models.RaceDirection import RaceDirection

from src.table_models.Canvas import Canvas
from src.windows.SocketThread import SocketThread
from src.packet_processing.variables import PLAYERS_LIST, COLUMN_SIZE_DICTIONARY, session
import src

# Database integration for export/import UI
try:
    from src.database import get_statistics, db_manager
    from src.export import CSVExporter, ParquetExporter
    from src.importers import CSVImporter
    DATABASE_UI_ENABLED = True
except ImportError:
    DATABASE_UI_ENABLED = False

# Phase 4 & 5: Advanced windows integration
try:
    from src.windows.analytics_window import AnalyticsWindow
    from src.windows.prediction_window import PredictionWindow
    from src.windows.strategy_window import StrategyWindow
    ADVANCED_WINDOWS_AVAILABLE = True
except ImportError:
    ADVANCED_WINDOWS_AVAILABLE = False
    print("[UI] Advanced windows not available - some features disabled")


class FixedSizeTabBar(QTabBar):
    def tabSizeHint(self, index):
        default_size = super().tabSizeHint(index)
        custom_widths = {
            0: 90,
            1: 110,
            2: 90,
            3: 180,
            4: 140,
            5: 70,
            6: 220,
            7: 200,
            8: 200
        }
        width = custom_widths.get(index, default_size.width())
        return QSize(width, default_size.height())

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Telemetry Application")
        self.resize(1080, 720)

        # Create menu bar with export/database controls
        self.create_menu_bar()

        self.socketThread = SocketThread()
        self.socketThread.data_received.connect(self.update_table)
        self.socketThread.start()

        self.main_layout = QVBoxLayout()

        self.index = 0

        self.menu = QListWidget()

        # Original menu items
        menu_items = ["Main", "Damage", "Laps", "Temperatures", "Map", "ERS & Fuel",
                     "Weather Forecast", "Packet Reception", "Race Director"]

        # Add new Phase 4 & 5 features if available
        if ADVANCED_WINDOWS_AVAILABLE:
            menu_items.extend(["Analytics 📊", "Predictions 🤖", "Strategy 🏁"])

        self.menu.addItems(menu_items)
        self.menu.setCurrentRow(self.index)

        self.mainModel = MainTableModel()
        self.damageModel = DamageTableModel()
        self.lapModel = LapTableModel()
        self.temperatureModel = TemperatureTableModel()
        self.mapCanvas = Canvas()
        self.ersAndFuelModel = ERSAndFuelTableModel()
        self.weatherForecastModel = WeatherForecastTableModel()
        self.packetReceptionTableModel = PacketReceptionTableModel(self)
        self.raceDirectorModel = RaceDirection()

        self.models = [
            self.mainModel,
            self.damageModel,
            self.lapModel,
            self.temperatureModel,
            self.mapCanvas,
            self.ersAndFuelModel,
            self.weatherForecastModel,
            self.packetReceptionTableModel,
            self.raceDirectorModel
        ]

        # Initialize advanced windows if available
        if ADVANCED_WINDOWS_AVAILABLE:
            try:
                print("[UI] Loading Analytics window...")
                self.analytics_window = AnalyticsWindow()
                print("[UI] Loading Predictions window...")
                self.prediction_window = PredictionWindow()
                print("[UI] Loading Strategy window...")
                self.strategy_window = StrategyWindow()

                # Add to models list
                self.models.extend([
                    self.analytics_window,
                    self.prediction_window,
                    self.strategy_window
                ])

                print("[UI] ✓ Advanced windows loaded successfully")
            except Exception as e:
                print(f"[UI] Warning: Failed to load advanced windows: {e}")
                # Note: Advanced windows failed to load but app continues with original features

        self.stack = QStackedWidget()
        self.menu.currentRowChanged.connect(self.on_row_changed)

        self.title_label = QLabel()

        self.create_layout()

        self.packet_reception_dict = [0 for _ in range(16)]
        self.last_update = 0

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)


        MainWindow.function_hashmap = [  # PacketId : (fonction, arguments)
            lambda packet : update_motion(packet),                                   # 0 : PacketMotion
            lambda packet : update_session(packet),                                  # 1 : PacketSession
            lambda packet : update_lap_data(packet),                                 # 2 : PacketLapData
            lambda packet : update_event(packet, self.raceDirectorModel),            # 3 : PacketEvent
            lambda packet : update_participants(packet),                             # 4 : PacketParticipants
            lambda packet : update_car_setups(packet),                               # 5 : PacketCarSetup
            lambda packet : update_car_telemetry(packet),                            # 6 : PacketCarTelemetry
            lambda packet : update_car_status(packet),                               # 7 : PacketCarStatus
            lambda packet : None,                                                    # 8 : PacketFinalClassification
            lambda packet : None,                                                    # 9 : PacketLobbyInfo
            lambda packet : update_car_damage(packet),                               # 10 : PacketCarDamage
            lambda packet : None,                                                    # 11 : PacketSessionHistory
            lambda packet : None,                                                    # 12 : PacketTyreSetsData
            lambda packet : update_motion_extended(packet),                          # 13 : PacketMotionExData
            lambda packet : None,                                                    # 14 : PacketTimeTrialData
            lambda packet : None                                                     # 15 : PacketLapPositions
        ]

    def create_menu_bar(self):
        """Create menu bar with Export and Database menus"""
        menubar = self.menuBar()

        # Export Menu
        export_menu = menubar.addMenu("Export")

        if DATABASE_UI_ENABLED:
            # Export to CSV
            export_csv_action = QAction("Export to CSV...", self)
            export_csv_action.setStatusTip("Export current session to CSV files")
            export_csv_action.triggered.connect(self.export_to_csv)
            export_menu.addAction(export_csv_action)

            # Export to Parquet
            export_parquet_action = QAction("Export to Parquet...", self)
            export_parquet_action.setStatusTip("Export current session to Parquet files (ML-optimized)")
            export_parquet_action.triggered.connect(self.export_to_parquet)
            export_menu.addAction(export_parquet_action)

            export_menu.addSeparator()

            # Import CSV
            import_csv_action = QAction("Import CSV...", self)
            import_csv_action.setStatusTip("Import historical CSV data into database")
            import_csv_action.triggered.connect(self.import_csv)
            export_menu.addAction(import_csv_action)
        else:
            # Database not available
            no_db_action = QAction("Database not available", self)
            no_db_action.setEnabled(False)
            export_menu.addAction(no_db_action)

        # Database Menu
        database_menu = menubar.addMenu("Database")

        if DATABASE_UI_ENABLED:
            # View Statistics
            stats_action = QAction("View Statistics", self)
            stats_action.setStatusTip("View database statistics")
            stats_action.triggered.connect(self.view_database_statistics)
            database_menu.addAction(stats_action)

            database_menu.addSeparator()

            # Export All Sessions
            export_all_action = QAction("Export All Sessions...", self)
            export_all_action.setStatusTip("Export all sessions to CSV/Parquet")
            export_all_action.triggered.connect(self.export_all_sessions)
            database_menu.addAction(export_all_action)
        else:
            # Database not available
            no_db_action = QAction("Database not available", self)
            no_db_action.setEnabled(False)
            database_menu.addAction(no_db_action)

        # Advanced Features Menu (Phase 5)
        if ADVANCED_WINDOWS_AVAILABLE:
            advanced_menu = menubar.addMenu("Advanced")

            # Anomaly Detection
            anomaly_action = QAction("Detect Anomalies...", self)
            anomaly_action.setStatusTip("Run anomaly detection on session data")
            anomaly_action.triggered.connect(self.run_anomaly_detection)
            advanced_menu.addAction(anomaly_action)

            # Multi-Session Learning
            learning_action = QAction("Train Track Model...", self)
            learning_action.setStatusTip("Train track-specific ML models")
            learning_action.triggered.connect(self.train_track_model)
            advanced_menu.addAction(learning_action)

            advanced_menu.addSeparator()

            # Live Alerts
            alerts_action = QAction("Strategy Alerts", self)
            alerts_action.setStatusTip("View live strategy alerts")
            alerts_action.triggered.connect(self.show_strategy_alerts)
            advanced_menu.addAction(alerts_action)

    def closeEvent(self, event):
        self.socketThread.stop()
        self.close()

    def create_layout(self):
        self.title_label.setFont(QFont("Segoe UI Emoji", 12))

        self.menu.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.menu.setMaximumWidth(150)
        self.menu.setFont(QFont("Segoe UI Emoji", 12))

        h_layout1 = QHBoxLayout()
        h_layout2 = QHBoxLayout()

        self.stack.addWidget(self.mainModel.table)
        self.stack.addWidget(self.damageModel.table)
        self.stack.addWidget(self.lapModel.table)
        self.stack.addWidget(self.temperatureModel.table)
        self.stack.addWidget(self.mapCanvas)
        self.stack.addWidget(self.ersAndFuelModel.table)
        self.stack.addWidget(self.weatherForecastModel.table)
        self.stack.addWidget(self.packetReceptionTableModel.table)
        self.stack.addWidget(self.raceDirectorModel)

        # Add advanced windows if available
        if ADVANCED_WINDOWS_AVAILABLE:
            self.stack.addWidget(self.analytics_window)
            self.stack.addWidget(self.prediction_window)
            self.stack.addWidget(self.strategy_window)
            print("[UI] ✓ Advanced windows added to stack")

        h_layout1.addWidget(self.title_label)
        h_layout2.addWidget(self.menu)
        h_layout2.addWidget(self.stack)

        self.main_layout.addLayout(h_layout1)
        self.main_layout.addLayout(h_layout2)

    def update_table(self, header, packet):
        MainWindow.function_hashmap[header.m_packet_id](packet)

        self.models[self.index].update()

        if header.m_packet_id == 1:
            self.title_label.setText(session.title_display())

        self.packet_reception_dict[header.m_packet_id] += 1
        if time.time() > self.last_update + 1:
            self.packetReceptionTableModel.update_each_second()
            self.packet_reception_dict = [0 for _ in range(16)]
            self.last_update = time.time()

    def on_row_changed(self, index):
        self.stack.setCurrentIndex(index)
        self.index = index

    def resizeEvent(self, event):
        src.packet_processing.variables.REDRAW_MAP = True
        super().resizeEvent(event)
        self.models[self.index].update()

    # Database Export/Import UI Handlers
    def export_to_csv(self):
        """Export current session to CSV files"""
        if not DATABASE_UI_ENABLED:
            return

        try:
            # Get recent sessions
            sessions = db_manager.get_recent_sessions(limit=10)
            if not sessions:
                QMessageBox.information(self, "No Data", "No sessions found in database.\n\nRun a session first to record data.")
                return

            # Let user select session
            session_items = [f"Session {s.id}: {s.track_name} - {s.session_type} ({s.created_at.strftime('%Y-%m-%d %H:%M')})"
                           for s in sessions]
            session_text, ok = QInputDialog.getItem(
                self, "Select Session", "Choose session to export:", session_items, 0, False
            )
            if not ok:
                return

            session_id = int(session_text.split(":")[0].split()[ 1])

            # Choose output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Export Directory", "exports"
            )
            if not output_dir:
                return

            # Export
            QMessageBox.information(self, "Exporting...", "Exporting session to CSV...\n\nThis may take a moment.")
            exporter = CSVExporter()
            exporter.export_session(session_id=session_id, output_dir=output_dir, include_telemetry=True)

            QMessageBox.information(
                self, "Export Complete",
                f"Session exported successfully!\n\nLocation: {output_dir}/\n\nFiles created:\n"
                f"- Laps CSV\n- Tyres CSV\n- Damage CSV\n- Telemetry CSV\n- Weather CSV"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export session:\n\n{str(e)}")

    def export_to_parquet(self):
        """Export current session to Parquet files (ML-optimized)"""
        if not DATABASE_UI_ENABLED:
            return

        try:
            # Get recent sessions
            sessions = db_manager.get_recent_sessions(limit=10)
            if not sessions:
                QMessageBox.information(self, "No Data", "No sessions found in database.\n\nRun a session first to record data.")
                return

            # Let user select session
            session_items = [f"Session {s.id}: {s.track_name} - {s.session_type} ({s.created_at.strftime('%Y-%m-%d %H:%M')})"
                           for s in sessions]
            session_text, ok = QInputDialog.getItem(
                self, "Select Session", "Choose session to export:", session_items, 0, False
            )
            if not ok:
                return

            session_id = int(session_text.split(":")[0].split()[1])

            # Choose output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Export Directory", "ml_data"
            )
            if not output_dir:
                return

            # Export
            QMessageBox.information(self, "Exporting...", "Exporting session to Parquet...\n\nThis may take a moment.")
            exporter = ParquetExporter()
            exporter.export_session(session_id=session_id, output_dir=output_dir, compression='snappy')

            QMessageBox.information(
                self, "Export Complete",
                f"Session exported successfully!\n\nLocation: {output_dir}/\n\n"
                f"Format: Parquet (5-10x smaller than CSV)\n"
                f"Compression: Snappy\n\nReady for ML training!"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export session:\n\n{str(e)}")

    def import_csv(self):
        """Import historical CSV data into database"""
        if not DATABASE_UI_ENABLED:
            return

        try:
            # Choose CSV file
            csv_file, _ = QFileDialog.getOpenFileName(
                self, "Select CSV File", "data", "CSV Files (*.csv)"
            )
            if not csv_file:
                return

            # Get track name and session type
            track_name, ok1 = QInputDialog.getText(
                self, "Track Name", "Enter track name (e.g., Melbourne):", text="Unknown"
            )
            if not ok1:
                return

            session_type_items = ["Practice", "Qualifying", "Race"]
            session_type, ok2 = QInputDialog.getItem(
                self, "Session Type", "Select session type:", session_type_items, 0, False
            )
            if not ok2:
                return

            # Import
            QMessageBox.information(self, "Importing...", "Importing CSV data...\n\nThis may take a moment.")
            importer = CSVImporter()
            session_id = importer.import_lap_csv(
                csv_path=csv_file,
                track_name=track_name,
                session_type=session_type
            )

            QMessageBox.information(
                self, "Import Complete",
                f"CSV imported successfully!\n\nSession ID: {session_id}\n"
                f"Imported {importer.imported_records} records"
            )
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import CSV:\n\n{str(e)}")

    def view_database_statistics(self):
        """View database statistics"""
        if not DATABASE_UI_ENABLED:
            return

        try:
            stats = get_statistics()
            stats_text = (
                f"📊 Database Statistics\n\n"
                f"Sessions: {stats['sessions']}\n"
                f"Laps: {stats['laps']}\n"
                f"Telemetry Snapshots: {stats['telemetry_snapshots']:,}\n"
                f"Tyre Data: {stats['tyre_data']}\n"
                f"Damage Events: {stats['damage_events']}\n"
                f"Pit Stops: {stats['pit_stops']}\n"
                f"Weather Samples: {stats['weather_samples']}\n\n"
                f"Database: f1_telemetry.db"
            )
            QMessageBox.information(self, "Database Statistics", stats_text)
        except Exception as e:
            QMessageBox.critical(self, "Statistics Error", f"Failed to retrieve statistics:\n\n{str(e)}")

    def export_all_sessions(self):
        """Export all sessions to CSV/Parquet"""
        if not DATABASE_UI_ENABLED:
            return

        try:
            # Get all sessions
            sessions = db_manager.get_recent_sessions(limit=100)
            if not sessions:
                QMessageBox.information(self, "No Data", "No sessions found in database.")
                return

            # Choose format
            format_items = ["CSV (Human-readable)", "Parquet (ML-optimized)"]
            format_choice, ok = QInputDialog.getItem(
                self, "Export Format", "Choose export format:", format_items, 0, False
            )
            if not ok:
                return

            # Choose output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Export Directory", "exports_all"
            )
            if not output_dir:
                return

            # Export all sessions
            QMessageBox.information(
                self, "Exporting...",
                f"Exporting {len(sessions)} sessions...\n\nThis may take several minutes."
            )

            if "CSV" in format_choice:
                exporter = CSVExporter()
                for s in sessions:
                    exporter.export_session(session_id=s.id, output_dir=output_dir, include_telemetry=False)
            else:
                exporter = ParquetExporter()
                session_ids = [s.id for s in sessions]
                exporter.export_for_ml_training(session_ids=session_ids, output_dir=output_dir)

            QMessageBox.information(
                self, "Export Complete",
                f"All {len(sessions)} sessions exported successfully!\n\nLocation: {output_dir}/"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export sessions:\n\n{str(e)}")

    # Advanced Features Handlers (Phase 5)
    def run_anomaly_detection(self):
        """Run anomaly detection on session"""
        if not ADVANCED_WINDOWS_AVAILABLE:
            return

        try:
            from src.advanced import AnomalyDetector

            # Get session ID
            sessions = db_manager.get_recent_sessions(limit=10)
            if not sessions:
                QMessageBox.information(self, "No Data", "No sessions found in database.")
                return

            session_items = [f"Session {s.id}: {s.track_name} - {s.session_type}" for s in sessions]
            session_text, ok = QInputDialog.getItem(
                self, "Select Session", "Choose session for anomaly detection:", session_items, 0, False
            )
            if not ok:
                return

            session_id = int(session_text.split(":")[0].split()[1])

            # Run detection
            QMessageBox.information(self, "Detecting...", "Running anomaly detection...\n\nThis may take a moment.")

            detector = AnomalyDetector()
            anomalies = detector.detect_all_anomalies(session_id, driver_index=0)
            report = detector.generate_report(anomalies)

            # Show report
            from PySide6.QtWidgets import QTextEdit, QDialog, QVBoxLayout
            dialog = QDialog(self)
            dialog.setWindowTitle("Anomaly Detection Report")
            dialog.resize(700, 600)

            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(report)
            text_edit.setFont(QFont("Courier", 10))

            layout = QVBoxLayout()
            layout.addWidget(text_edit)
            dialog.setLayout(layout)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Anomaly detection failed:\n\n{str(e)}")

    def train_track_model(self):
        """Train track-specific model"""
        if not ADVANCED_WINDOWS_AVAILABLE:
            return

        try:
            from src.advanced import MultiSessionLearning

            # Get track name
            track_name, ok = QInputDialog.getText(
                self, "Track Name", "Enter track name (e.g., Monaco, Spa):", text="Monaco"
            )
            if not ok or not track_name:
                return

            # Get sessions
            sessions = db_manager.get_recent_sessions(limit=20)
            session_ids = [s.id for s in sessions]

            if not session_ids:
                QMessageBox.information(self, "No Data", "No sessions found.")
                return

            # Train
            QMessageBox.information(
                self, "Training...",
                f"Training {track_name} model on {len(session_ids)} sessions...\n\nThis may take a moment."
            )

            learner = MultiSessionLearning()
            result = learner.train_track_specific_model(
                track_name=track_name,
                session_ids=session_ids,
                model_type='tyre_wear'
            )

            QMessageBox.information(
                self, "Training Complete",
                f"Track model trained successfully!\n\n"
                f"Track: {result['track']}\n"
                f"Samples: {result['samples']}\n"
                f"Test R²: {result['test_r2']:.3f}\n"
                f"Model saved to: {result['model_path']}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed:\n\n{str(e)}")

    def show_strategy_alerts(self):
        """Show live strategy alerts"""
        if not ADVANCED_WINDOWS_AVAILABLE:
            return

        try:
            from src.alerts import StrategyAlertsEngine

            # Mock current state for demo
            current_state = {
                'current_lap': 25,
                'current_position': 5,
                'tyre_age': 20,
                'avg_wear': 65.0,
                'total_laps': 50
            }

            engine = StrategyAlertsEngine()
            alerts = engine.check_all_alerts(current_state)

            if not alerts:
                QMessageBox.information(self, "Strategy Alerts", "✓ No active alerts\n\nEverything looks good!")
                return

            # Format alerts
            alert_text = "🚨 LIVE STRATEGY ALERTS\n\n"
            for alert in alerts:
                alert_text += f"{alert['severity']}: {alert['message']}\n"
                alert_text += f"→ {alert['recommendation']}\n\n"

            QMessageBox.information(self, "Strategy Alerts", alert_text)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to check alerts:\n\n{str(e)}")




