"""
Startup Dialog - Choose between Real-time UDP or Manual Data Upload
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                                QLabel, QFileDialog, QRadioButton, QButtonGroup,
                                QWidget, QGroupBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from pathlib import Path


class StartupDialog(QDialog):
    """Dialog shown at startup to choose data source"""

    mode_selected = Signal(str, str)  # Signal: (mode, file_path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("F1 Telemetry Application - Choose Data Source")
        self.setModal(True)
        self.setMinimumSize(600, 400)

        self.selected_mode = None
        self.selected_file = None

        self.setup_ui()

    def setup_ui(self):
        """Create the UI"""
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Title
        title = QLabel("🏁 F1 Telemetry Application")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Choose your data source to get started")
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Options container
        options_layout = QHBoxLayout()
        options_layout.setSpacing(20)

        # Option 1: Real-time UDP
        realtime_group = self.create_option_box(
            "📡 Real-time F1 Game",
            "Connect to F1 25 (or F1 22/23/24) via UDP\n\n"
            "• Requires game running\n"
            "• Live telemetry data\n"
            "• Port 20777\n"
            "• Real-time updates",
            "realtime"
        )
        options_layout.addWidget(realtime_group)

        # Option 2: Manual data upload
        manual_group = self.create_option_box(
            "📁 Manual Data Upload",
            "Load recorded telemetry from file\n\n"
            "• Upload .bin recording\n"
            "• Replay past sessions\n"
            "• Loop playback\n"
            "• Offline testing",
            "manual"
        )
        options_layout.addWidget(manual_group)

        layout.addLayout(options_layout)

        layout.addSpacing(20)

        # File selection for manual mode (initially hidden)
        self.file_selection_widget = QWidget()
        file_layout = QVBoxLayout()

        file_label = QLabel("Select telemetry recording file:")
        file_label.setStyleSheet("font-weight: bold;")
        file_layout.addWidget(file_label)

        file_btn_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #666; padding: 5px; border: 1px solid #ddd; border-radius: 3px;")
        file_btn_layout.addWidget(self.file_path_label, 1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)
        file_btn_layout.addWidget(self.browse_btn)

        file_layout.addLayout(file_btn_layout)
        self.file_selection_widget.setLayout(file_layout)
        self.file_selection_widget.setVisible(False)

        layout.addWidget(self.file_selection_widget)

        layout.addStretch()

        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)

        self.start_btn = QPushButton("Start Application")
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 30px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_btn.clicked.connect(self.start_application)
        buttons_layout.addWidget(self.start_btn)

        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def create_option_box(self, title, description, mode):
        """Create an option box with radio button"""
        group_box = QGroupBox()
        group_box.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin-top: 10px;
            }
            QGroupBox:hover {
                border-color: #4CAF50;
            }
        """)

        layout = QVBoxLayout()

        # Radio button with title
        radio = QRadioButton(title)
        radio_font = QFont()
        radio_font.setPointSize(14)
        radio_font.setBold(True)
        radio.setFont(radio_font)
        radio.toggled.connect(lambda checked: self.on_mode_changed(mode, checked))
        layout.addWidget(radio)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #555; margin-left: 25px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        group_box.setLayout(layout)

        # Store radio button reference
        if not hasattr(self, 'radio_buttons'):
            self.radio_buttons = {}
        self.radio_buttons[mode] = radio

        # Make the whole box clickable
        group_box.mousePressEvent = lambda event: radio.setChecked(True)

        return group_box

    def on_mode_changed(self, mode, checked):
        """Handle mode selection change"""
        if checked:
            self.selected_mode = mode

            # Show/hide file selection based on mode
            if mode == "manual":
                self.file_selection_widget.setVisible(True)
                # Enable start button only if file is selected
                self.start_btn.setEnabled(self.selected_file is not None)
            else:
                self.file_selection_widget.setVisible(False)
                # Enable start button for real-time mode
                self.start_btn.setEnabled(True)

    def browse_file(self):
        """Open file browser to select telemetry file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Telemetry Recording",
            str(Path.home()),
            "Telemetry Files (*.bin);;All Files (*.*)"
        )

        if file_path:
            self.selected_file = file_path
            self.file_path_label.setText(Path(file_path).name)
            self.file_path_label.setStyleSheet("color: #000; padding: 5px; border: 1px solid #4CAF50; border-radius: 3px;")
            self.start_btn.setEnabled(True)

    def start_application(self):
        """Emit signal with selected mode and accept dialog"""
        if self.selected_mode:
            self.mode_selected.emit(self.selected_mode, self.selected_file or "")
            self.accept()
