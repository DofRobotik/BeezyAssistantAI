import sys
import asyncio
import os
from typing import Optional
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QScrollArea,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
import qasync

# Path ayarları (Aynı kalabilir)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from assistant_ptt import AssistantApp
except ImportError:
    AssistantApp = None


class VoiceAssistantUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.assistant_app: Optional[AssistantApp] = None
        self.current_assistant_bubble = None
        self.last_user_bubble = None  # Son kullanıcı balonunu takip et
        self.is_recording = False
        self.setup_ui()
        QTimer.singleShot(100, lambda: asyncio.create_task(self.init_assistant()))

    def setup_ui(self):
        self.setWindowTitle("Beezy AI Assistant")
        self.setGeometry(100, 100, 500, 700)
        self.setStyleSheet("background-color: #2c3e50;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title = QLabel("Beezy AI")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: white; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: transparent; border: none;")
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()
        self.scroll_area.setWidget(self.chat_widget)
        layout.addWidget(self.scroll_area)

        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("color: #bdc3c7; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.mic_button = QPushButton("🎤")
        self.mic_button.setFixedSize(80, 80)
        self.mic_button.setFont(QFont("Segoe UI", 30))
        self.mic_button.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; border-radius: 40px; }"
            "QPushButton:hover { background-color: #2980b9; }"
        )
        self.mic_button.clicked.connect(self.toggle_recording)
        self.mic_button.setEnabled(False)

        btn_container = QHBoxLayout()
        btn_container.addStretch()
        btn_container.addWidget(self.mic_button)
        btn_container.addStretch()
        layout.addLayout(btn_container)

    async def init_assistant(self):
        callbacks = {
            "on_listening_started": self.on_listening,
            "on_processing_started": self.on_processing,
            "on_transcription_done": self.on_transcription_complete,  # Yeni Callback
            "on_ready": self.on_ready,
            "on_error": self.on_error,
            "on_response_start": self.on_response_start,
            "on_response_chunk": self.on_response_chunk,
            "on_response_end": self.on_response_end,
        }
        if AssistantApp:
            self.assistant_app = AssistantApp(callbacks=callbacks)
            asyncio.create_task(self.assistant_app.run())

    def toggle_recording(self):
        if not self.assistant_app:
            return
        if not self.is_recording:
            self.assistant_app.start_listening()
        else:
            asyncio.create_task(self.assistant_app.stop_listening())

    # --- CALLBACKS ---
    def on_listening(self):
        self.is_recording = True
        self.status_label.setText("Listening...")
        self.mic_button.setStyleSheet(
            "background-color: #e74c3c; border-radius: 40px; color: white;"
        )
        self.mic_button.setText("⏹")

    def on_processing(self, text_placeholder):
        """STT başladığında 'Transcribing...' balonunu ekler."""
        self.is_recording = False
        self.status_label.setText("Processing...")
        self.mic_button.setEnabled(False)
        self.mic_button.setStyleSheet("background-color: #95a5a6; border-radius: 40px;")
        self.mic_button.setText("⏳")
        # Kullanıcı balonunu ekle ve referansını sakla
        self.last_user_bubble = self.add_bubble(text_placeholder, is_user=True)

    def on_transcription_complete(self, real_text):
        """STT bittiğinde balondaki metni günceller."""
        if self.last_user_bubble:
            self.last_user_bubble.setText(real_text)

    def on_ready(self):
        self.is_recording = False
        self.status_label.setText("Ready")
        self.mic_button.setEnabled(True)
        self.mic_button.setStyleSheet(
            "background-color: #3498db; border-radius: 40px; color: white;"
        )
        self.mic_button.setText("🎤")

    def on_response_start(self):
        self.current_assistant_bubble = self.add_bubble("", is_user=False)

    def on_response_chunk(self, text):
        if self.current_assistant_bubble:
            current = self.current_assistant_bubble.text()
            self.current_assistant_bubble.setText(current + text)
            self.scroll_to_bottom()

    def on_response_end(self):
        self.current_assistant_bubble = None

    def on_error(self, msg):
        self.status_label.setText(f"Error: {msg}")
        # Hata olursa 'Transcribing...' balonunu güncelle
        if self.last_user_bubble and self.last_user_bubble.text() == "Transcribing...":
            self.last_user_bubble.setText(f"Error: {msg}")
        self.on_ready()

    # --- HELPERS ---
    def add_bubble(self, text, is_user):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setFont(QFont("Segoe UI", 12))
        lbl.setContentsMargins(15, 10, 15, 10)
        lbl.setMaximumWidth(350)

        if is_user:
            lbl.setStyleSheet(
                "background-color: #3498db; color: white; border-radius: 15px;"
            )
            align = Qt.AlignRight
        else:
            lbl.setStyleSheet(
                "background-color: #ecf0f1; color: #2c3e50; border-radius: 15px;"
            )
            align = Qt.AlignLeft

        container = QHBoxLayout()
        if is_user:
            container.addStretch()
        container.addWidget(lbl)
        if not is_user:
            container.addStretch()

        w = QWidget()
        w.setLayout(container)
        self.chat_layout.addWidget(w)
        self.scroll_to_bottom()
        return lbl

    def scroll_to_bottom(self):
        sb = self.scroll_area.verticalScrollBar()
        sb.setValue(sb.maximum())


def main():
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    win = VoiceAssistantUI()
    win.show()
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
