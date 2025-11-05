import sys
import asyncio
import time
from typing import Optional
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                              QHBoxLayout, QWidget, QTextEdit, QLabel, 
                              QPushButton, QFrame, QScrollArea)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QBrush
import qasync
import traceback
# --- YENİ ---
# Gerçek AssistantApp'ı içe aktar
try:
    from src.assistant import AssistantApp
except ImportError:
    print("HATA: 'assistant.py' dosyası bulunamadı.")
    AssistantApp = None # Hata durumunda sahte sınıf

class VoiceActivityWidget(QWidget):
    """Animated widget showing voice activity"""
    def __init__(self):
        super().__init__()
        self.setFixedSize(100, 100)
        self.is_listening = False
        self.is_speaking = False
        self.animation_frame = 0
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        
    def start_listening(self):
        self.is_listening = True
        self.is_speaking = False
        if not self.timer.isActive():
            self.timer.start(50)  # 20 FPS
        
    def start_speaking(self):
        self.is_listening = False
        self.is_speaking = True
        if not self.timer.isActive():
            self.timer.start(100)  # 10 FPS
        
    def stop_activity(self):
        self.is_listening = False
        self.is_speaking = False
        self.timer.stop()
        self.update()
        
    def update_animation(self):
        self.animation_frame = (self.animation_frame + 1) % 60
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center_x, center_y = self.width() // 2, self.height() // 2
        
        if self.is_listening:
            # Pulsing circle for listening
            base_radius = 30
            pulse = abs(30 - (self.animation_frame % 60)) / 30.0
            radius = base_radius + (pulse * 10)
            
            painter.setBrush(QBrush(QColor(52, 152, 219, int(100 + pulse * 100))))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center_x - radius, center_y - radius, 
                              radius * 2, radius * 2)
            
            # Inner circle
            painter.setBrush(QBrush(QColor(52, 152, 219)))
            painter.drawEllipse(center_x - 15, center_y - 15, 30, 30)
            
        elif self.is_speaking:
            # Sound waves for speaking
            painter.setPen(QColor(46, 204, 113))
            painter.setBrush(QBrush(QColor(46, 204, 113)))
            
            # Center circle
            painter.drawEllipse(center_x - 8, center_y - 8, 16, 16)
            
            # Animated sound waves
            for i in range(3):
                wave_radius = 20 + (i * 15)
                opacity = int(255 * (0.8 - (self.animation_frame % 20) / 25.0))
                if opacity > 0:
                    painter.setPen(QColor(46, 204, 113, opacity))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(center_x - wave_radius, center_y - wave_radius,
                                      wave_radius * 2, wave_radius * 2)
        else:
            # Idle state
            painter.setBrush(QBrush(QColor(149, 165, 166)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center_x - 20, center_y - 20, 40, 40)


class ConversationBubble(QFrame):
    """Chat bubble for conversation display"""
    def __init__(self, text: str, is_user: bool = True):
        super().__init__()
        self.setMaximumWidth(400)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        # --- YENİ ---
        # Metin etiketini daha sonra güncellemek için sakla
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setFont(QFont("Segoe UI", 11))
        
        if is_user:
            # User bubble (right side, blue)
            self.setStyleSheet("""
                QFrame {
                    background-color: #3498db;
                    border-radius: 18px;
                    color: white;
                }
            """)
            self.text_label.setAlignment(Qt.AlignRight)
        else:
            # Assistant bubble (left side, gray)
            self.setStyleSheet("""
                QFrame {
                    background-color: #ecf0f1;
                    border-radius: 18px;
                    color: #2c3e50;
                }
            """)
            self.text_label.setAlignment(Qt.AlignLeft)
            
        layout.addWidget(self.text_label)
        self.setLayout(layout)

    # --- YENİ METOT ---
    def set_text(self, text: str):
        """Baloncuğun metnini tamamen değiştirir."""
        self.text_label.setText(text)

    # --- YENİ METOT ---
    def append_text(self, chunk: str):
        """Baloncuğun mevcut metnine ekleme yapar."""
        current_text = self.text_label.text()
        self.text_label.setText(current_text + chunk)
        # Metin değiştikçe QFrame'in yeniden boyutlanmasını sağla
        self.adjustSize()


class StatusWidget(QFrame):
    """Status display widget"""
    def __init__(self):
        super().__init__()
        self.setFixedHeight(60)
        self.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 10px;
                color: white;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 20, 0)
        
        self.status_label = QLabel("Ready to listen...")
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        
    def set_status(self, status: str, color: str = "#ecf0f1"):
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"color: {color};")


class MockAssistant:
    """Mock assistant for demo purposes (streaming'i simüle eder)"""
    def __init__(self, ui):
        self.ui = ui
        self.running = True
        
    async def stop(self):
        self.running = False
        print("Mock Assistant durduruluyor...")
        
    async def run(self):
        await asyncio.sleep(2)
        if not self.running: return

        self.ui.on_listening_started()
        await asyncio.sleep(3)
        if not self.running: return

        self.ui.on_processing_started("Hello, how are you today?")
        await asyncio.sleep(2)
        if not self.running: return

        # --- YENİ: Streaming simülasyonu ---
        self.ui.on_assistant_response_start() # Boş baloncuk yarat
        demo_text = "Hello! I'm doing great, thank you for asking. How can I help you today?"
        for char in demo_text:
            if not self.running: return
            self.ui.on_assistant_response_chunk(char)
            await asyncio.sleep(0.05) # Token-by-token efekti
        self.ui.on_assistant_response_end()
        # --- BİTTİ ---

        await asyncio.sleep(2)
        if not self.running: return
        self.ui.on_ready()


class AssistantUI(QMainWindow):
    """Main UI window for the voice assistant"""
    
    def __init__(self):
        super().__init__()
        # --- YENİ ---
        # AssistantApp veya MockAssistant'ı saklamak için
        self.assistant_app: Optional[AssistantApp | MockAssistant] = None

        self.current_assistant_bubble: Optional[ConversationBubble] = None

        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Voice Assistant")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(600, 400)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QScrollArea {
                border: none;
                background-color: #2c3e50;
            }
            QScrollBar:vertical {
                background-color: #34495e;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #7f8c8d;
                border-radius: 4px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Voice Assistant")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #ecf0f1; margin: 10px;")
        
        # Voice activity widget
        activity_layout = QHBoxLayout()
        activity_layout.addStretch()
        self.voice_activity = VoiceActivityWidget()
        activity_layout.addWidget(self.voice_activity)
        activity_layout.addStretch()
        
        # Status widget
        self.status_widget = StatusWidget()
        
        # Conversation area
        self.conversation_scroll = QScrollArea()
        self.conversation_scroll.setWidgetResizable(True)
        self.conversation_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.conversation_widget = QWidget()
        self.conversation_layout = QVBoxLayout()
        self.conversation_layout.setSpacing(10)
        self.conversation_layout.addStretch()
        self.conversation_widget.setLayout(self.conversation_layout)
        self.conversation_scroll.setWidget(self.conversation_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Assistant")
        self.start_button.setFont(QFont("Segoe UI", 12))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #219a52;
            }
        """)
        self.start_button.clicked.connect(lambda: asyncio.create_task(self.start_assistant()))
        
        self.stop_button = QPushButton("Stop Assistant")
        self.stop_button.setFont(QFont("Segoe UI", 12))
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.stop_button.clicked.connect(self.stop_assistant)
        self.stop_button.setEnabled(False)
        
        # Demo mode button
        self.demo_button = QPushButton("Demo Mode")
        self.demo_button.setFont(QFont("Segoe UI", 12))
        self.demo_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #7d3c98;
            }
        """)
        self.demo_button.clicked.connect(lambda: asyncio.create_task(self.start_demo()))
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.demo_button)
        
        # Add all widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addLayout(activity_layout)
        main_layout.addWidget(self.status_widget)
        main_layout.addWidget(self.conversation_scroll, 1)  # Give it most space
        main_layout.addLayout(button_layout)
        
        central_widget.setLayout(main_layout)
        
    def add_conversation_bubble(self, text: str, is_user: bool = True) -> Optional[ConversationBubble]:
        """
        Add a conversation bubble to the chat.
        --- YENİ ---
        Returns:
            ConversationBubble: Yaratılan baloncuk nesnesi (eğer asistana aitse)
        """
        bubble = ConversationBubble(text, is_user)
        
        # Create container for alignment
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        if is_user:
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        else:
            container_layout.addWidget(bubble)
            container_layout.addStretch()
            
        container_widget = QWidget()
        container_widget.setLayout(container_layout)
        
        # Remove the stretch at the end temporarily
        self.conversation_layout.takeAt(self.conversation_layout.count() - 1)
        
        # Add the new bubble
        self.conversation_layout.addWidget(container_widget)
        
        # Add stretch back at the end
        self.conversation_layout.addStretch()
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
        
        # --- YENİ ---
        # Eğer asistan baloncuğu ise, referansını döndür
        if not is_user:
            return bubble
        return None
        
    def scroll_to_bottom(self):
        scrollbar = self.conversation_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def on_listening_started(self):
        """Called when the assistant starts listening"""
        self.voice_activity.start_listening()
        self.status_widget.set_status("Listening (Seni Dinliyorum)...", "#3498db")
        
    def on_processing_started(self, user_text: str):
        """Called when processing user input"""
        self.voice_activity.stop_activity()
        self.status_widget.set_status("Processing (Düşünüyorum)...", "#f39c12")
        self.add_conversation_bubble(user_text, is_user=True)
        
    # def on_speaking_started(self, assistant_text: str):
    #     """Called when assistant starts speaking"""
        
    #     # --- YENİ DEBUG PRINT ---
    #     print(f"[DEBUG] UI 'on_speaking_started' TETİKLENDİ: '{assistant_text}'")
        
    #     self.voice_activity.start_speaking()
    #     self.status_widget.set_status("Speaking (Konuşuyorum)...", "#2ecc71")
    #     self.add_conversation_bubble(assistant_text, is_user=False)
        
    def on_assistant_response_start(self):
        """LLM'in ilk token'ı geldiğinde çağrılır."""
        self.voice_activity.start_speaking()
        self.status_widget.set_status("Speaking (Konuşuyorum)...", "#2ecc71")
        
        # Boş bir baloncuk yarat ve referansını sakla
        self.current_assistant_bubble = self.add_conversation_bubble("", is_user=False)
    
    def on_assistant_response_chunk(self, chunk: str):
        """LLM'den yeni bir 'chunk' geldiğinde çağrılır."""
        if self.current_assistant_bubble:
            self.current_assistant_bubble.append_text(chunk)
            # Metin eklendikçe aşağı kaydır
            QTimer.singleShot(10, self.scroll_to_bottom)
    
    def on_assistant_response_end(self):
        """LLM yanıtı bittiğinde çağrılır."""
        self.current_assistant_bubble = None # Referansı temizle
        # Not: TTS hala devam ediyor olabilir, o yüzden 'on_ready' 
        # asistanın kendisinden (işi bitince) gelmeli.
    
    def on_ready(self):
        """Called when assistant is ready for next input"""
        self.voice_activity.stop_activity()
        self.status_widget.set_status("Ready to listen (Dinlemeye Hazırım)...", "#ecf0f1")
        
    def on_error(self, error_msg: str):
        """Called when an error occurs"""
        self.status_widget.set_status(f"Error: {error_msg}", "#e74c3c")
        self.voice_activity.stop_activity()
        
        # Hata durumunda butonları sıfırla
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.demo_button.setEnabled(True)

    # --- METOT GÜNCELLENDİ ---
    async def start_assistant(self):
        """Start the real voice assistant"""
        if not AssistantApp:
            self.on_error("AssistantApp içe aktarılamadı.")
            return
            
        try:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.demo_button.setEnabled(False)
            self.status_widget.set_status("Starting assistant...", "#f39c12")
            
            # --- GÜNCELLENMİŞ CALLBACK'LER ---
            callbacks = {
                "on_listening_started": self.on_listening_started,
                "on_processing_started": self.on_processing_started,
                "on_ready": self.on_ready,
                "on_error": self.on_error,
                # Yeni streaming callback'leri
                "on_response_start": self.on_assistant_response_start,
                "on_response_chunk": self.on_assistant_response_chunk,
                "on_response_end": self.on_assistant_response_end
            }
            
            self.assistant_app = AssistantApp(callbacks=callbacks)
            await self.assistant_app.run()
            
        except Exception as e:
            # --- YENİ ---
            # Gerçek hata izini terminale yazdır!
            print("!!!!! ASİSTAN BAŞLATILIRKEN KRİTİK HATA !!!!!")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            self.status_widget.set_status(f"Başlatma Hatası: {str(e)}", "#e74c3c")
            self.on_error(f"Başlatma Hatası: {str(e)}") # Butonları sıfırlar
        
        finally:
            # Asistanın durduğundan emin ol ve butonları sıfırla
            # (Hata oluşsa bile burası çalışır ve "Stopped" yazar)
            self.status_widget.set_status("Stopped", "#95a5a6")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.demo_button.setEnabled(True)
            self.voice_activity.stop_activity()
            self.assistant_app = None

            
    async def start_demo(self):
        """Start demo mode"""
        try:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.demo_button.setEnabled(False)
            self.status_widget.set_status("Starting demo...", "#f39c12")
            
            # Create mock assistant
            self.assistant_app = MockAssistant(self)
            
            # Start the demo
            await self.assistant_app.run()
            
        except Exception as e:
            self.status_widget.set_status(f"Demo error: {str(e)}", "#e74c3c")
        
        finally:
            # Demo bittiğinde butonları sıfırla
            self.status_widget.set_status("Demo Finished", "#95a5a6")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.demo_button.setEnabled(True)
            self.voice_activity.stop_activity()
            self.assistant_app = None

    # --- METOT GÜNCELLENDİ ---
    def stop_assistant(self):
        """Stop the voice assistant or demo"""
        if self.assistant_app:
            self.status_widget.set_status("Stopping...", "#95a5a6")
            # assistant_app.stop() metodunu asenkron olarak tetikle
            # Bu, assistant.run() döngüsünü sonlandıracak
            asyncio.create_task(self.assistant_app.stop())
        
        # Butonlar 'start_assistant' veya 'start_demo' 
        # metodlarının 'finally' bloğunda sıfırlanacak.
        self.stop_button.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the UI
    window = AssistantUI()
    window.show()
    
    # Use qasync event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    with loop:
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            print("Application terminated by user")
        finally:
            loop.close()


if __name__ == "__main__":
    main()