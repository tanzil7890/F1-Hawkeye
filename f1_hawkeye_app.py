import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from src.windows.main_window import MainWindow
from src.windows.startup_dialog import StartupDialog
from src.windows.PlaybackThread import PlaybackThread

# Database initialization
try:
    from src.database import initialize_database
    from src.database.data_writer import telemetry_writer
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("[Telemetry] Database module not available - running in memory-only mode")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    with open("style.css", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)

    # Initialize database and start writer thread
    if DATABASE_AVAILABLE:
        try:
            print("[Database] Initializing database...")
            initialize_database()
            telemetry_writer.start()
            print("[Database] ✓ Database ready (f1_telemetry.db)")
            print("[Database] ✓ Writer thread started")
        except Exception as e:
            print(f"[Database] Warning: Failed to initialize ({e})")
            print("[Database] Running in memory-only mode")
            DATABASE_AVAILABLE = False

    # Show startup dialog to choose mode
    startup = StartupDialog()

    # Store mode and file path
    class AppState:
        """Container for application state"""
        selected_mode = None
        selected_file = None
        playback_thread = None

    def on_mode_selected(mode, file_path):
        """Handle mode selection from startup dialog"""
        AppState.selected_mode = mode
        AppState.selected_file = file_path

        if mode == "manual" and file_path:
            # Start playback thread for manual data
            AppState.playback_thread = PlaybackThread(
                file_path=file_path,
                port=20777,
                host='127.0.0.1',
                speed=1.0,
                loop=True
            )

            # Connect signals
            AppState.playback_thread.playback_started.connect(
                lambda: print(f"▶️  Playback started: {file_path}")
            )
            AppState.playback_thread.playback_progress.connect(
                lambda packets, timestamp: print(f"📤 Sent {packets} packets (@ {timestamp:.1f}s)")
                if packets % 200 == 0 else None
            )
            AppState.playback_thread.playback_finished.connect(
                lambda total: print(f"🔁 Loop complete: {total} packets")
            )
            AppState.playback_thread.playback_error.connect(
                lambda error: QMessageBox.critical(None, "Playback Error", error)
            )

            # Start playback in background
            AppState.playback_thread.start()
            print(f"\n📁 Manual Data Mode: Playing {file_path}")
            print(f"🔁 Looping playback on port 20777\n")

        else:
            print(f"\n📡 Real-time Mode: Listening on port 20777")
            print(f"🎮 Connect F1 game with UDP telemetry enabled\n")

    startup.mode_selected.connect(on_mode_selected)

    # Show dialog and check result
    if startup.exec() == StartupDialog.Accepted:
        # Create and show main window
        window = MainWindow()
        window.show()

        # Store playback thread reference in window for cleanup
        if AppState.playback_thread:
            window.playback_thread = AppState.playback_thread

            # Stop playback when window closes
            def cleanup():
                if hasattr(window, 'playback_thread') and window.playback_thread:
                    print("\n🛑 Stopping playback...")
                    window.playback_thread.stop()

                # Stop database writer
                if DATABASE_AVAILABLE:
                    print("[Database] Stopping writer thread...")
                    telemetry_writer.stop(timeout=5.0)
                    print("[Database] ✓ Writer stopped")

            window.destroyed.connect(cleanup)
        else:
            # No playback thread, but still need database cleanup
            def cleanup():
                if DATABASE_AVAILABLE:
                    print("[Database] Stopping writer thread...")
                    telemetry_writer.stop(timeout=5.0)
                    print("[Database] ✓ Writer stopped")

            window.destroyed.connect(cleanup)

        sys.exit(app.exec())
    else:
        # User cancelled - cleanup database
        if DATABASE_AVAILABLE:
            print("[Database] Stopping writer thread...")
            telemetry_writer.stop(timeout=5.0)
            print("[Database] ✓ Writer stopped")

        print("Application cancelled by user")
        sys.exit(0)