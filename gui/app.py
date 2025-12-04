import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLineEdit, QPushButton, QLabel, QStackedWidget,
    QFileDialog, QMessageBox, QTextEdit
)

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QFont, QPalette, QBrush, QPixmap, QIcon
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from output import predict


class EntryPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top spacer to position title below the upper net
        layout.addSpacing(90)
        
        # Title with modern football styling
        title = QLabel("FairPlayAi")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Arial Black", 48, QFont.Weight.ExtraBold)
        title.setFont(title_font)
        title.setStyleSheet("""
            QLabel {
                color: #decc09;
                background-color: rgba(0, 0, 0, 0.7);
                padding: 20px 40px;
                border-radius: 15px;
                border: 3px solid #decc09;
                text-shadow: 0 0 10px #decc09, 0 0 20px #decc09;
            }
        """)
        
        title_container = QHBoxLayout()
        title_container.addStretch()
        title_container.addWidget(title)
        title_container.addStretch()
        layout.addLayout(title_container)
        
        # Large spacer
        layout.addStretch(2)
        
        # Subtitle/tagline
        subtitle = QLabel("Analyze Football Fouls with AI Precision")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont("Segoe UI", 18)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 0.6);
                padding: 12px 30px;
                border-radius: 25px;
                font-weight: 500;
            }
        """)
        
        subtitle_container = QHBoxLayout()
        subtitle_container.addStretch()
        subtitle_container.addWidget(subtitle)
        subtitle_container.addStretch()
        layout.addLayout(subtitle_container)
        
        layout.addStretch(1)
        
        # Input section with modern design
        input_container = QWidget()
        input_container.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.75);
                border-radius: 20px;
                padding: 30px;
            }
        """)
        
        input_layout = QVBoxLayout()
        input_layout.setSpacing(20)
        
        # Label for input
        input_label = QLabel("Upload your clip:")
        input_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                background: transparent;
                padding: 0;
            }
        """)
        input_layout.addWidget(input_label)
        
        # Input and button in horizontal layout
        input_row = QHBoxLayout()
        input_row.setSpacing(15)
        
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Enter video path or click Browse...")
        self.video_input.setMinimumHeight(55)
        self.video_input.setStyleSheet("""
            QLineEdit {
                padding: 15px 20px;
                font-size: 16px;
                border: 2px solid #decc09;
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                color: #1a1a1a;
            }
            QLineEdit:focus {
                border: 3px solid #decc09;
                background-color: white;
            }
        """)
        
        self.browse_btn = QPushButton("📁 BROWSE")
        self.browse_btn.setMinimumHeight(55)
        self.browse_btn.setMinimumWidth(140)
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4A90E2, stop:1 #357ABD);
                border: none;
                color: white;
                font-weight: bold;
                font-size: 15px;
                padding: 15px 25px;
                border-radius: 12px;
                text-transform: uppercase;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5BA3F5, stop:1 #4A90E2);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #357ABD, stop:1 #2868A8);
            }
        """)
        
        self.confirm_btn = QPushButton("ANALYZE ▶")
        self.confirm_btn.setMinimumHeight(55)
        self.confirm_btn.setMinimumWidth(160)
        self.confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #decc09, stop:1 #00CC33);
                border: none;
                color: #000000;
                font-weight: bold;
                font-size: 16px;
                padding: 15px 30px;
                border-radius: 12px;
                text-transform: uppercase;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #33FF66, stop:1 #decc09);
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00CC33, stop:1 #009922);
            }
        """)
        
        input_row.addWidget(self.video_input, stretch=3)
        input_row.addWidget(self.browse_btn, stretch=1)
        input_row.addWidget(self.confirm_btn, stretch=1)
        
        input_layout.addLayout(input_row)
        input_container.setLayout(input_layout)
        
        # Add input container to main layout with margins
        input_outer = QHBoxLayout()
        input_outer.setContentsMargins(60, 0, 60, 60)
        input_outer.addWidget(input_container)
        layout.addLayout(input_outer)
        
        self.setLayout(layout)


class VideoPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = ""
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.init_ui()
    
    def init_ui(self):
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Top bar with back button and title
        top_bar = QHBoxLayout()
        
        # Back button - Cyberpunk style
        self.back_btn = QPushButton("◄ BACK")
        self.back_btn.setMinimumHeight(40)
        self.back_btn.setMinimumWidth(100)
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0a0a, stop:1 #1a1a1a);
                border: 2px solid #00ffff;
                color: #00ffff;
                font-family: 'Courier New', monospace;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 16px;
                border-radius: 5px;
                text-shadow: 0 0 8px #00ffff;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:1 #2a2a2a);
                border: 2px solid #00ffff;
            }
            QPushButton:pressed {
                background: #000000;
                border: 2px solid #00aaaa;
                color: #00aaaa;
            }
        """)
        
        # Title - Cyberpunk style
        title = QLabel("MODEL OUTPUT")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Courier New", 20, QFont.Weight.ExtraBold)
        title.setFont(title_font)
        title.setStyleSheet("""
            QLabel {
                color: #decc09;
                background-color: rgba(0, 0, 0, 0.8);
                padding: 8px 20px;
                border: 2px solid #decc09;
                border-radius: 5px;
                text-shadow: 0 0 10px #decc09, 0 0 20px #decc09;
                letter-spacing: 2px;
            }
        """)
        
        top_bar.addWidget(self.back_btn)
        top_bar.addStretch()
        top_bar.addWidget(title)
        top_bar.addStretch()
        top_bar.addSpacing(100)
        
        layout.addLayout(top_bar)
        
        # Video widget with border - Cyberpunk style
        video_container = QWidget()
        video_container.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                border: 3px solid #00ffff;
                border-radius: 5px;
            }
        """)
        
        video_container_layout = QVBoxLayout()
        video_container_layout.setContentsMargins(5, 5, 5, 5)
        
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(200)
        self.video_widget.setMaximumHeight(220)
        self.video_widget.setStyleSheet("""
            QVideoWidget {
                background-color: #000000;
            }
        """)
        
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.media_player.setVideoOutput(self.video_widget)
        
        video_container_layout.addWidget(self.video_widget)
        video_container.setLayout(video_container_layout)
        
        layout.addWidget(video_container)
        
        # Video controls - Cyberpunk style
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        controls_layout.addStretch()
        
        self.play_btn = QPushButton("▶ PLAY")
        self.play_btn.setMinimumHeight(35)
        self.play_btn.setMinimumWidth(90)
        self.play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00ff00, stop:1 #00aa00);
                border: 2px solid #00ff00;
                color: #000000;
                font-family: 'Courier New', monospace;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 12px;
                border-radius: 5px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00ff66, stop:1 #00cc00);
            }
            QPushButton:pressed {
                background: #009900;
            }
        """)
        
        self.pause_btn = QPushButton("⏸ PAUSE")
        self.pause_btn.setMinimumHeight(35)
        self.pause_btn.setMinimumWidth(90)
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffaa00, stop:1 #ff6600);
                border: 2px solid #ffaa00;
                color: #000000;
                font-family: 'Courier New', monospace;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 12px;
                border-radius: 5px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffcc00, stop:1 #ff8800);
            }
            QPushButton:pressed {
                background: #ff6600;
            }
        """)
        
        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setMinimumWidth(90)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff0066, stop:1 #cc0000);
                border: 2px solid #ff0066;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 12px;
                border-radius: 5px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff0088, stop:1 #ee0000);
            }
            QPushButton:pressed {
                background: #990000;
            }
        """)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Card section - Cyberpunk style
        card_label = QLabel("CARD:")
        card_label.setStyleSheet("""
            QLabel {
                color: #decc09;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                text-shadow: 0 0 8px #decc09;
                letter-spacing: 2px;
                background-color: rgba(0, 0, 0, 0.7);
                padding: 6px 12px;
                border: 2px solid #decc09;
                border-radius: 5px;
            }
        """)
        layout.addWidget(card_label)
        
        # Card text box
        self.card_textbox = QTextEdit()
        self.card_textbox.setPlaceholderText("Card prediction will appear here...")
        self.card_textbox.setReadOnly(True)
        self.card_textbox.setMinimumHeight(50)
        self.card_textbox.setMaximumHeight(50)
        self.card_textbox.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.9);
                border: 2px solid #00ffff;
                border-radius: 5px;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                padding: 6px;
                selection-background-color: #00ffff;
                selection-color: #000000;
            }
        """)
        layout.addWidget(self.card_textbox)
        
        # Action section - Cyberpunk style
        action_label = QLabel("ACTION:")
        action_label.setStyleSheet("""
            QLabel {
                color: #decc09;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                text-shadow: 0 0 8px #decc09;
                letter-spacing: 2px;
                background-color: rgba(0, 0, 0, 0.7);
                padding: 6px 12px;
                border: 2px solid #decc09;
                border-radius: 5px;
            }
        """)
        layout.addWidget(action_label)
        
        # Action text box
        self.action_textbox = QTextEdit()
        self.action_textbox.setPlaceholderText("Action classification will appear here...")
        self.action_textbox.setReadOnly(True)
        self.action_textbox.setMinimumHeight(50)
        self.action_textbox.setMaximumHeight(50)
        self.action_textbox.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.9);
                border: 2px solid #00ffff;
                border-radius: 5px;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                padding: 6px;
                selection-background-color: #00ffff;
                selection-color: #000000;
            }
        """)
        layout.addWidget(self.action_textbox)
        
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Connect control buttons
        self.play_btn.clicked.connect(self.media_player.play)
        self.pause_btn.clicked.connect(self.media_player.pause)
        self.stop_btn.clicked.connect(self.media_player.stop)
    
    def set_video_path(self, path):
        self.video_path = path
        self.media_player.setSource(QUrl.fromLocalFile(path))
        print(f"Video path set to: {path}")
        
        # Run prediction and update textboxes
        try:
            predictions = predict(path)
            self.card_textbox.setText(predictions[0])
            self.action_textbox.setText(predictions[1])
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.card_textbox.setText("Error: Could not analyze video")
            self.action_textbox.setText("Error: Could not analyze video")
    
    def set_card_output(self, text):
        """Update the card textbox with prediction results"""
        self.card_textbox.setText(text)
    
    def set_action_output(self, text):
        """Update the action textbox with classification results"""
        self.action_textbox.setText(text)


class FairPlayAiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("FairPlayAi")
        self.setMinimumSize(600, 600)
        self.setMaximumSize(600, 600)
        self.resize(600, 600)
        
        # Set application icon
        icon_path = 'icon.png'
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print("Warning: app_icon.png not found. Using default icon.")
        
        # Create stacked widget for page navigation
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create pages
        self.entry_page = EntryPage()
        self.video_page = VideoPage()
        
        # Set background image for entry page
        self.set_entry_background()
        
        # Add pages to stack
        self.stacked_widget.addWidget(self.entry_page)
        self.stacked_widget.addWidget(self.video_page)
        
        # Connect page switching to update background
        self.stacked_widget.currentChanged.connect(self.on_page_changed)
        
        # Connect confirm button to page transition
        self.entry_page.confirm_btn.clicked.connect(self.go_to_video_page)
        
        # Connect browse button to file dialog
        self.entry_page.browse_btn.clicked.connect(self.browse_video_file)
        
        # Connect back button to return to entry page
        self.video_page.back_btn.clicked.connect(self.go_to_entry_page)
        
        # Allow Enter key to submit
        self.entry_page.video_input.returnPressed.connect(self.go_to_video_page)
    
    def set_entry_background(self):
        """Set the soccer net background for entry page"""
        background = QPixmap('aaaa.png')
        
        if not background.isNull():
            palette = QPalette()
            scaled_background = background.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            palette.setBrush(QPalette.ColorRole.Window, QBrush(scaled_background))
            self.setPalette(palette)
            self.setAutoFillBackground(True)
        else:
            self.setStyleSheet("background-color: #1a1a1a;")
            print("Warning: soccer_goal_net.png not found. Using fallback background.")
    
    def set_video_background(self):
        """Set the cyberpunk background for video page"""
        background = QPixmap('cyberpunk_background.png')
        
        if not background.isNull():
            palette = QPalette()
            scaled_background = background.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            palette.setBrush(QPalette.ColorRole.Window, QBrush(scaled_background))
            self.setPalette(palette)
            self.setAutoFillBackground(True)
        else:
            # Fallback to dark cyberpunk background
            self.setStyleSheet("QMainWindow { background-color: #0a0a0a; }")
            print("Warning: cyberpunk_background.png not found. Using fallback background.")
    
    def on_page_changed(self, index):
        """Update background when switching pages"""
        if index == 0:  # Entry page
            self.set_entry_background()
        else:  # Video page - Cyberpunk background
            self.set_video_background()
    
    def browse_video_file(self):
        """Open file dialog to select a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*.*)"
        )
        
        if file_path:
            self.entry_page.video_input.setText(file_path)
    
    def go_to_video_page(self):
        video_path = self.entry_page.video_input.text().strip()
        
        if not video_path:
            QMessageBox.warning(
                self,
                "No Video Selected",
                "Please enter or select a video file path."
            )
            return
        
        # Check if file exists
        if not os.path.exists(video_path):
            QMessageBox.critical(
                self,
                "File Not Found",
                f"The file does not exist:\n{video_path}"
            )
            return
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(video_path):
            QMessageBox.critical(
                self,
                "Invalid Selection",
                "Please select a valid video file."
            )
            return
        
        # Check if it has a valid video extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']
        file_ext = os.path.splitext(video_path)[1].lower()
        
        if file_ext not in valid_extensions:
            reply = QMessageBox.warning(
                self,
                "Invalid Video Format",
                f"The selected file may not be a valid video format.\n\nSupported formats: {', '.join(valid_extensions)}\n\nDo you want to continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # All validation passed, proceed to video page
        self.video_page.set_video_path(video_path)
        VIDEO_PATH = video_path
        self.stacked_widget.setCurrentWidget(self.video_page)
    
    def go_to_entry_page(self):
        """Return to the entry page and stop video playback"""
        self.video_page.media_player.stop()
        self.stacked_widget.setCurrentWidget(self.entry_page)
        # Force background refresh
        self.set_entry_background()


def main():
    app = QApplication(sys.argv)
    window = FairPlayAiApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()