import sys
import os
import time
import queue
import threading
import re
import asyncio
import base64
import cv2
import numpy as np
from datetime import datetime
import audioop
# --- 3rd Party Imports ---
from dotenv import load_dotenv
import requests
import pyaudio
from deepface import DeepFace

# --- SDK Imports ---
from elevenlabs import ElevenLabs, AudioFormat, CommitStrategy, RealtimeAudioOptions, RealtimeEvents
from google import genai
from google.genai import types
from googleapiclient.discovery import build
from ddgs import DDGS
from mem0 import Memory

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'robot_head'))
sys.path.append(os.path.join(current_dir, 'api_nonlive'))
# --- Custom Modules (User Provided) ---
# Bu dosyaların proje klasöründe olduğunu varsayıyoruz
try:
    # Başındaki noktaları (.) sildik:
    from robot_head.detectors.face_detector import FaceDetector
    from robot_head.sensors.realsense_cam import RealSenseCamera
    from robot_head.pan_tilt_module.pan_tilt_module import PanTiltController
    from robot_head.config import COM_PORT
    
    print("✅ Donanım modülleri başarıyla yüklendi.")

except ImportError as e:
    print(f"⚠️ HATA DETAYI: {e}") 
    print("⚠️ UYARI: Donanım modülleri bulunamadı. Mock modunda çalışıyor.")
    
    COM_PORT = "COM3"
    FaceDetector = None
    RealSenseCamera = None
    PanTiltController = None

try:
    # Burada da noktayı kaldırdık
    from api_nonlive.iot import AmrLoungeClass
except ImportError:
    print("⚠️ IoT modülü bulunamadı.")
    AmrLoungeClass = None

# --- PySide6 Imports ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextBrowser, QGraphicsDropShadowEffect, 
    QInputDialog, QMessageBox, QFrame, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QObject
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings


load_dotenv()

# --- AYARLAR ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ses Ayarları
FORMAT = pyaudio.paInt16
CHANNELS = 1
MIC_RATE = 16000
TTS_RATE = 22050
CHUNK = 1024
ROBOT_ID = "amr-1"
FACE_DB_PATH = "face_db"

if not os.path.exists(FACE_DB_PATH):
    os.makedirs(FACE_DB_PATH)

# ============================================================================
# 1. BELLEK YÖNETİCİSİ (DİNAMİK KULLANICI DESTEKLİ)
# ============================================================================
class MemoryManager:
    def __init__(self, initial_user_id="Guest"):
        self.user_id = initial_user_id
        self.gemini_api_key = GEMINI_API_KEY
        
        vector_store_config = {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "beezy_permanent_storage",
                "embedding_model_dims": 768, 
            }
        }
        config = {
            "vector_store": vector_store_config,
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-2.5-flash",
                    "api_key": self.gemini_api_key,
                    "temperature": 0.1
                }
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": self.gemini_api_key,
                    "embedding_dims": 768
                }
            }
        }
        
        print("🧠 Mem0 Hafıza Sistemi Başlatılıyor...")
        try:
            self.memory = Memory.from_config(config)
            print("✅ Mem0 Modülü Hazır.")
        except Exception as e:
            print(f"❌ Mem0 Başlatma Hatası: {e}")
            self.memory = None

    def set_user_id(self, new_user_id):
        """Kullanıcı değiştiğinde çağrılır."""
        if self.user_id != new_user_id:
            print(f"🔄 Hafıza Bağlamı Değiştiriliyor: {self.user_id} -> {new_user_id}")
            self.user_id = new_user_id
            return True
        return False

    def get_relevant_context(self, query):
        if not self.memory: return ""
        try:
            results = self.memory.search(query, user_id=self.user_id, limit=5, threshold=0.35)
            if not results: return ""
            formatted = []
            res_list = results.get("results") if isinstance(results, dict) else results
            if res_list:
                for m in res_list:
                    text = m.get("memory", m.get("text", str(m)))
                    formatted.append(f"- {text}")
            final_text = "\n".join(formatted)
            return final_text
        except Exception as e:
            print(f"⚠️ Hafıza Arama Hatası: {e}")
            return ""

    def add_interaction(self, user_text, ai_text):
        if not self.memory: return
        messages = [{"role": "user", "content": user_text}, {"role": "assistant", "content": ai_text}]
        extraction_prompt = (
            "You are a memory extraction system. Extract facts about the User. "
            "PRIORITIZE: Name, Likes, Dislikes, Location requests. "
            "Ignore generic greetings."
        )
        try:
            self.memory.add(messages, user_id=self.user_id, prompt=extraction_prompt)
        except Exception as e:
            print(f"🔥 Hafıza Ekleme Hatası: {e}")

# ============================================================================
# 2. VISION WORKER (GÖRÜNTÜ İŞLEME VE TAKİP)
# ============================================================================
class VisionWorker(QThread):
    change_pixmap_signal = Signal(QImage)
    face_detected_signal = Signal(bool) # Yüz var mı yok mu
    face_recognized_signal = Signal(str) # Tanınan kişinin ID'si
    system_log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.register_next_face = False
        self.new_face_name = None
        self.cam = None
        self.pt = None
        self.face_detector = None

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_register_mode(self, name):
        """Arayüzden 'Beni Hatırla'ya basılınca tetiklenir."""
        self.new_face_name = name
        self.register_next_face = True

    def run(self):
        if not RealSenseCamera or not FaceDetector:
            self.system_log_signal.emit("Kamera veya FaceDetector modülü eksik. Vision Thread durduruldu.")
            return

        self.cam = RealSenseCamera()
        self.face_detector = FaceDetector()
        
        try:
            self.pt = PanTiltController(port=COM_PORT, pan_offset=-4.0, tilt_offset=5.0, debug=False)
            self.system_log_signal.emit(f"Pan-Tilt Bağlandı: {COM_PORT}")
            time.sleep(2.0)
        except Exception as e:
            self.system_log_signal.emit(f"Pan-Tilt Hatası: {e}")
            self.pt = None

        tracked_face_center = None
        last_face_time = time.time()
        no_face_timeout = 2.0
        
        # DeepFace optimizasyonu için sayaç
        frame_count = 0
        recognition_interval = 30 # Her 30 karede bir tanı (FPS düşmemesi için)
        current_user_id = "Unknown"

        while self._run_flag:
            color_img, depth_frame = self.cam.get_frames()
            if color_img is None:
                continue

            frame_count += 1
            h, w, _ = color_img.shape
            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            detections = self.face_detector.detect_faces(color_img)

            # --- YÜZ YOKSA ---
            if len(detections) == 0:
                self.face_detected_signal.emit(False)
                if self.pt and (time.time() - last_face_time) > no_face_timeout:
                    self.pt.send_angles(80, 90) # Home position
                    tracked_face_center = None
            else:
                self.face_detected_signal.emit(True)
                last_face_time = time.time()

                # --- TRACKING LOGIC (Mainv2'den alındı) ---
                faces_info = []
                for x1, y1, x2, y2, conf in detections:
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    depth = depth_frame.get_distance(cx, cy)
                    if depth <= 0: continue
                    faces_info.append((x1, y1, x2, y2, conf, cx, cy, depth))

                if not faces_info: continue

                selected_face = None
                
                # Tracking algoritması (En yakını veya son takip edileni seç)
                if tracked_face_center is not None:
                    min_dist = float('inf')
                    for face in faces_info:
                        _, _, _, _, _, cx, cy, _ = face
                        dist = np.hypot(cx - tracked_face_center[0], cy - tracked_face_center[1])
                        if dist < min_dist:
                            min_dist = dist
                            selected_face = face
                    if min_dist > 200: 
                        selected_face = min(faces_info, key=lambda f: f[-1]) # En yakın derinlik
                        tracked_face_center = None
                else:
                    selected_face = min(faces_info, key=lambda f: f[-1])

                # --- SEÇİLEN YÜZ İŞLEMLERİ ---
                if selected_face:
                    x1, y1, x2, y2, conf, cx, cy, depth = selected_face
                    tracked_face_center = (cx, cy)

                    # 1. Pan-Tilt Hareketi
                    if self.pt:
                        X = (cx - intr.ppx) / intr.fx * depth
                        Y = (cy - intr.ppy) / intr.fy * depth
                        Z = depth
                        try:
                            self.pt.send_xyz(X, Y, Z)
                        except: pass

                    # 2. Yüz Kaydetme (Butona basıldıysa)
                    if self.register_next_face and self.new_face_name:
                        face_roi = color_img[y1:y2, x1:x2]
                        if face_roi.size > 0:
                            save_path = os.path.join(FACE_DB_PATH, f"{self.new_face_name}.jpg")
                            cv2.imwrite(save_path, face_roi)
                            self.system_log_signal.emit(f"✅ Yüz Kaydedildi: {self.new_face_name}")
                            
                            # Veritabanını güncellemek için mevcut pickle'ı sil (DeepFace reload etsin diye)
                            pkl_path = os.path.join(FACE_DB_PATH, "representations_vgg_face.pkl") # Modele göre değişir
                            if os.path.exists(pkl_path): os.remove(pkl_path)
                            
                            # Anlık olarak tanınan kişiyi güncelle
                            current_user_id = self.new_face_name
                            self.face_recognized_signal.emit(current_user_id)
                            
                        self.register_next_face = False
                        self.new_face_name = None

                    # 3. Yüz Tanıma (DeepFace) - Periyodik
                    if frame_count % recognition_interval == 0 and not self.register_next_face:
                        try:
                            # Tüm görseli değil sadece yüzü gönder
                            face_roi = color_img[y1:y2, x1:x2]
                            if face_roi.shape[0] > 20 and face_roi.shape[1] > 20:
                                # Not: enforce_detection=False çünkü zaten detect ettik
                                dfs = DeepFace.find(
                                    img_path=face_roi, 
                                    db_path=FACE_DB_PATH, 
                                    model_name="VGG-Face", 
                                    enforce_detection=False, 
                                    silent=True
                                )
                                
                                if len(dfs) > 0 and not dfs[0].empty:
                                    # Dosya yolundan ID çıkarma (face_db/ahmet.jpg -> ahmet)
                                    full_path = dfs[0].iloc[0]['identity']
                                    found_id = os.path.basename(full_path).split('.')[0]
                                    
                                    if found_id != current_user_id:
                                        current_user_id = found_id
                                        self.face_recognized_signal.emit(current_user_id)
                                else:
                                    if current_user_id != "Unknown":
                                        current_user_id = "Unknown"
                                        self.face_recognized_signal.emit("Unknown")
                        except Exception as e:
                            print(f"DeepFace Hatası: {e}")

                    # Çizim
                    color = (0, 255, 0) if current_user_id != "Unknown" else (0, 0, 255)
                    cv2.rectangle(color_img, (x1, y1), (x2, y2), color, 2)
                    label = f"ID: {current_user_id}"
                    cv2.putText(color_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Görüntüyü GUI'ye gönder
            rgb_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)

        # Temizlik
        if self.cam: self.cam.stop()
        if self.pt: self.pt.close()

# ============================================================================
# 3. AI WORKER (SES VE ZEKA) - FLUSH FIX (KUYRUK BOŞALTMA DÜZELTMESİ)
# ============================================================================
class AIWorker(QObject):
    text_received = Signal(str, str)
    status_changed = Signal(str)
    video_found = Signal(str, str)
    
    def __init__(self):
        super().__init__()
        self.user_id = "Guest"
        self.is_recording = False
        self.running = True
        self.commit_needed = False
        self.stop_generation = False
        self.is_ai_speaking = False

        self.mic_queue = queue.Queue(maxsize=100) 
        self.audio_out_queue = queue.Queue()
        
        self.eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self.ddgs = DDGS()
        self.p = pyaudio.PyAudio()

        self.memory_manager = MemoryManager(initial_user_id=self.user_id)
        self.chat_history = []
        self.transcript_parts = []
        
        self.setup_gemini_config()

        # Thread'leri başlat
        threading.Thread(target=self._audio_player_loop, daemon=True).start()
        threading.Thread(target=self._mic_producer_loop, daemon=True).start()
        threading.Thread(target=self._start_async_loop, daemon=True).start()

    def update_active_user(self, new_user_id):
        if new_user_id == "Unknown":
            pass
        else:
            changed = self.memory_manager.set_user_id(new_user_id)
            if changed:
                self.user_id = new_user_id
                self.chat_history = [] 
                self.text_received.emit("System", f"👋 Merhaba {new_user_id}, hafızan yüklendi!")

    def setup_gemini_config(self):
        self.all_iot_device_codes = []
        if AmrLoungeClass:
            try:
                iot_service_url = "10.10.10.244"
                iot_port = 3001
                self.iot = AmrLoungeClass(iot_service_url, iot_port)
                self.light_device_map = {}
                for group_index, devices in self.iot._AmrLoungeClass__lounge_place.items():
                    for place_index, device in enumerate(devices):
                        code = device["code"]
                        self.all_iot_device_codes.append(code)
                        self.light_device_map[code] = {"group": group_index, "index": place_index}
            except Exception as e:
                self.all_iot_device_codes = []
                self.light_device_map = {}

        self.ROS_NAV_ENDPOINT = "http://10.10.190.14:8000/navigate"
        self.stations = [
            {"name": "station_a", "property": "Food Court area."},
            {"name": "station_b", "property": "Restrooms area."},
            {"name": "station_c", "property": "Fun room area."},
            {"name": "station_d", "property": "A garment shop."},
            {"name": "station_e", "property": "A tech shop."},
        ]
        self.station_names = [s["name"] for s in self.stations]
        
        self.tools = [
             types.Tool(function_declarations=[
                types.FunctionDeclaration(name="find_youtube_video", description="Search youtube video", parameters={"type": "object", "properties": {"search_query": {"type": "string"}}, "required": ["search_query"]}),
                types.FunctionDeclaration(name="navigate_to_station", description="Navigation", parameters={"type": "object", "properties": {"target_station": {"type": "string","enum":self.station_names}}, "required": ["target_station"]}),
                types.FunctionDeclaration(name="control_iot_device", description="IoT Control", parameters={"type": "object", "properties": {"target_device_code": {"type": "string","enum":self.all_iot_device_codes}, "action": {"type": "string", "enum": ["turn_on", "turn_off"]}}, "required": ["target_device_code", "action"]}),
                types.FunctionDeclaration(name="search_web", description="Search web", parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
                types.FunctionDeclaration(name="sense_of_response",description="Sense of the given response",parameters={"type":"object","properties":{"emotion":{"type":"string","enum":["happy","sad"]}},"required":["emotion"]})
            ])
        ]
        
        self.gemini_config = types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=("You are Beezy... (Promptun geri kalanı aynı)"),
            thinking_config=types.ThinkingConfig(thinking_budget=256)
        )

    def _mic_producer_loop(self):
        """Mikrofon verisini okur ve görsel olarak seviyesini basar."""
        # Sistem varsayılan mikrofonunu açar
        try:
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=MIC_RATE, input=True, frames_per_buffer=CHUNK)
            print("🎙️ Mikrofon Donanımı Aktif")
        except Exception as e:
            print(f"❌ MİKROFON AÇILAMADI: {e}")
            return

        while self.running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # --- SES SEVİYESİ TESTİ (VU METER) ---
                # Sadece kayıt yaparken göster
                if self.is_recording:
                    rms = audioop.rms(data, 2)  # Sesin gücünü ölç
                    # Görsel çubuk oluştur (Gürültü eşiği: 300)
                    if rms > 300:
                        bar = "|" * int(rms / 100) 
                        sys.stdout.write(f"\r🔊 Seviye: {bar[:50]}") # Max 50 karakter
                        sys.stdout.flush()
                    
                    if not self.mic_queue.full():
                        self.mic_queue.put(data)
                
                time.sleep(0.001) 
            except Exception as e:
                # print(f"Mic Error: {e}") # Konsolu kirletmesin diye kapattım
                time.sleep(0.1)

    def interrupt_playback(self):
        with self.audio_out_queue.mutex:
            self.audio_out_queue.queue.clear()
        self.is_ai_speaking = False
        self.stop_generation = True

    def _start_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._elevenlabs_stt_handler())

    async def _elevenlabs_stt_handler(self):
        retry_count = 0
        while self.running:
            disconnect_event = asyncio.Event()
            
            try:
                if retry_count > 0:
                    await asyncio.sleep(min(2 ** (retry_count - 1), 5))

                print("🔌 ElevenLabs STT Bağlanıyor...")
                connection = await self.eleven_client.speech_to_text.realtime.connect(
                    RealtimeAudioOptions(
                        model_id="scribe_v2_realtime",
                        language_code="tr",
                        audio_format=AudioFormat.PCM_16000,
                        sample_rate=16000,
                        commit_strategy=CommitStrategy.MANUAL,
                        include_timestamps=False,
                    )
                )
                print("✅ STT Bağlantısı Başarılı.")
                self.status_changed.emit("Sistem Hazır")
                retry_count = 0

                async def send_audio_loop():
                    last_activity_time = time.time()
                    silence_base64 = base64.b64encode(b'\x00' * 1024).decode('utf-8')

                    while not disconnect_event.is_set() and self.running:
                        try:
                            current_time = time.time()
                            
                            # 1. ASİSTAN KONUŞUYORSA (Heartbeat gönder)
                            if self.is_ai_speaking:
                                # Mikrofon kuyruğunu boşalt (yankı olmasın)
                                while not self.mic_queue.empty(): 
                                    try: self.mic_queue.get_nowait()
                                    except: pass
                                
                                if (current_time - last_activity_time) > 2.0:
                                    await connection.send({"audio_base_64": silence_base64, "sample_rate": 16000})
                                    last_activity_time = current_time
                                await asyncio.sleep(0.1)
                                continue

                            # -----------------------------------------------------------
                            # DÜZELTME BURADA: "is_recording" kontrolünü kaldırdık.
                            # Kuyrukta veri varsa, kayıt bitmiş olsa bile GÖNDERMELİSİNİZ.
                            # -----------------------------------------------------------
                            
                            # 2. KUYRUKTAKİ SESİ GÖNDER (Flush)
                            sent_audio = False
                            while not self.mic_queue.empty():
                                chunk = self.mic_queue.get_nowait()
                                if chunk:
                                    chunk_base64 = base64.b64encode(chunk).decode('utf-8')
                                    await connection.send({"audio_base_64": chunk_base64, "sample_rate": 16000})
                                    last_activity_time = current_time
                                    sent_audio = True
                            
                            if sent_audio:
                                continue # Veri gönderdik, döngü başına dön

                            # 3. COMMIT (Ancak kuyruk boşaldıktan sonra)
                            if self.commit_needed and self.mic_queue.empty():
                                print("\n⚡ Commit Gönderiliyor... (Buffer Boş)")
                                await connection.commit()
                                self.commit_needed = False
                                last_activity_time = current_time

                            # 4. IDLE (Boşta Bekleme)
                            elif (current_time - last_activity_time) > 3.0:
                                await connection.send({"audio_base_64": silence_base64, "sample_rate": 16000})
                                last_activity_time = current_time
                            
                            else:
                                await asyncio.sleep(0.005)

                        except Exception as e:
                            print(f"Send Loop Hatası: {e}")
                            disconnect_event.set()
                            break

                send_task = asyncio.create_task(send_audio_loop())

                def on_partial_transcript(data):
                    txt = data.get('text', '') if isinstance(data, dict) else getattr(data, 'text', '')
                    if txt and self.is_recording:
                        sys.stdout.write(f"\r📝 {txt}")
                        sys.stdout.flush()

                def on_committed_transcript(data):
                    txt = data.get('text', '') if isinstance(data, dict) else getattr(data, 'text', '')
                    if txt:
                        print(f"\n✅ Algılandı: {txt}")
                        self.transcript_parts.append(txt)

                def on_error(error):
                    print(f"STT Error: {error}")
                    disconnect_event.set()

                def on_close():
                    disconnect_event.set()

                connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial_transcript)
                connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed_transcript)
                connection.on(RealtimeEvents.ERROR, on_error)
                connection.on(RealtimeEvents.CLOSE, on_close)

                await disconnect_event.wait()
                if send_task: send_task.cancel()
                try: await connection.close()
                except: pass

            except Exception as e:
                print(f"Bağlantı Hatası (Retry): {e}")
                retry_count += 1
                if not self.running: break

    @Slot()
    def start_recording_session(self):
        self.transcript_parts = []
        self.is_recording = True
        self.commit_needed = False
        self.status_changed.emit("Dinleniyor...")
        print("\n🎤 [KAYIT BAŞLADI]")

    @Slot()
    def stop_recording_session(self):
        if not self.is_recording: return
        
        # 1. Kaydı durdur (Producer artık veri atmayacak)
        self.is_recording = False
        
        # 2. Kuyrukta işlenmemiş ses kalmışsa temizle (İsteğe bağlı: Son heceyi kaybetmemek için bunu kapatabilirsin)
        # Genellikle tuşu bıraktıktan sonraki sesi istemeyiz, o yüzden temizlemek güvenlidir.
        # with self.mic_queue.mutex:
        #    self.mic_queue.queue.clear()
        
        # 3. Async döngüye "Commit yap" emri ver
        self.commit_needed = True 
        
        self.status_changed.emit("İşleniyor...")
        print("\n🛑 [KAYIT BİTTİ - Commit İsteği Gönderildi]")
        
        threading.Timer(1.0, self._finalize_and_process).start()

    def _finalize_and_process(self):
        full_text = " ".join(self.transcript_parts).strip()
        print(f"📝 İşlenen Metin (Final): '{full_text}'")
        
        if full_text:
            self.text_received.emit("User", full_text)
            threading.Thread(target=self._process_llm_response, args=(full_text,)).start()
            self.transcript_parts = [] 
        else:
            self.status_changed.emit("Ses algılanamadı.")
            print("⚠️ UYARI: ElevenLabs boş transkript döndürdü. Mikrofon ses seviyesini kontrol edin.")

    def _audio_player_loop(self):
        stream = None
        while self.running:
            try:
                if stream is None:
                    stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=TTS_RATE, output=True)
                
                # Kuyruktan veri al
                try:
                    data = self.audio_out_queue.get(timeout=0.1)
                except queue.Empty:
                    # Veri bittiyse ve kuyruk boşsa konuşma bitmiştir
                    if self.is_ai_speaking and self.audio_out_queue.empty():
                        self.is_ai_speaking = False
                    continue

                if data:
                    self.is_ai_speaking = True  # <--- Ses çalarken mikrofonu susturmak için işaretle
                    stream.write(data)
                    self.audio_out_queue.task_done()
            except:
                pass
    
    # ... Diğer metodlar (perform_web_search, search_youtube, navigation, iot vb.) AYNEN KALACAK ...
    def search_youtube(self, query):
        try:
            youtube = build("youtube", "v3", developerKey=GOOGLE_SEARCH_API_KEY)
            req = youtube.search().list(part="snippet", q=query, maxResults=1, type="video", videoEmbeddable="true")
            res = req.execute()
            if res["items"]:
                vid = res["items"][0]["id"]["videoId"]
                title = res["items"][0]["snippet"]["title"]
                embed_url = f"https://www.youtube-nocookie.com/embed/{vid}?autoplay=1"
                self.video_found.emit(embed_url, title)
                return f"Video bulundu: {title}"
        except: pass
        return "Video bulunamadı."
    
    def perform_web_search(self, query):
        try:
            results = self.ddgs.text(query, max_results=2,region="tr-tr",language="tr-tr")
            if results: return results[0]['body']
        except: pass
        return "Arama hatası."

    def _process_llm_response(self, user_text):
        try:
            relevant_memories = self.memory_manager.get_relevant_context(user_text)
            if relevant_memories:
                augmented_user_text = (
                    f"<history_context>\n"
                    f"The following are past interactions and facts about the user. "
                    f"DO NOT use them as current commands. Only use them to personalize the conversation:\n"
                    f"{relevant_memories}\n"
                    f"</history_context>\n\n"
                    f"CURRENT USER REQUEST: {user_text}"
                )
            else:
                augmented_user_text = user_text

            print(f"🤖 LLM'e Giden Prompt:\n{augmented_user_text}")

            chat = self.gemini_client.chats.create(model="gemini-2.5-flash", history=self.chat_history, config=self.gemini_config)
            response_stream = chat.send_message_stream(augmented_user_text)
            buffer = ""
            full_response_text = ""
            link_to_send_at_end = None
            
            for chunk in response_stream:
                if self.stop_generation:
                    print("🛑 LLM Üretimi Barge-in ile kesildi.")
                    break

                if chunk.text:
                    text_part = chunk.text
                    buffer += text_part
                    full_response_text += text_part
                    self.text_received.emit("AI_Partial", text_part)
                    if re.search(r'[.!?:\n]', text_part):
                        self._stream_tts(buffer)
                        buffer = ""

                if chunk.function_calls:
                    for fc in chunk.function_calls:
                        fname = fc.name
                        args = fc.args
                        response_data = None
                        if fname == "sense_of_response":
                            emotion = args.get("emotion")
                            if emotion:
                                print(f"--- 🤖 MODEL DUYGUSU: {emotion} ---")
                                response_data = {
                                    "success": True,
                                    "message": f"Sonuç duygu:{emotion}",
                                }

                        elif fname == "search_web":
                            search_query = args.get("query")
                            result = self.perform_web_search(search_query)
                            print(f"Web aranıyor...{search_query} sonucu aranıyor...")
                            response_data = {
                                "success":True,
                                "message":f"Sonuç web araması:{result}"
                            }  
                        elif fname == "find_youtube_video":
                            query = args.get("search_query")
                            if not query:
                                raise ValueError(
                                    "Arama sorgusu (search_query) eksik."
                                )

                            print(
                                f"--- 🔎 YouTube API Araması (Tool Call) Başlatılıyor: '{query}' ---"
                            )
                            # Bizim özel YouTube API fonksiyonumuzu çağır
                            result = self.search_youtube(query)

                            if result and result[0]:
                                embed_url, video_title = result
                                print(
                                    f"--- 🔗 YOUTUBE BULUNDU (GÖNDERİM BEKLETİLİYOR): {video_title} ---"
                                )

                                # Turn bitiminde göndermek için sakla (Çakışma Önleyici)
                                link_to_send_at_end = (embed_url, video_title)

                                response_data = {
                                    "success": True,
                                    "video_title": video_title,
                                    "message": "Video bulundu. Konuşma bitiminde gösterilecek.",
                                }
                            else:
                                response_data = {
                                    "success": False,
                                    "message": f"Video bulunamadı: {query}",
                                }
                        elif fname == "navigate_to_station":
                            target = args.get("target_station")

                            print(
                                f"✅ Navigasyon: {target} hedefine yönlendiriliyor..."
                            )

                            # Komutu doğrudan çalıştır
                            success, message = self.execute_navigation_command(target)

                            response_data = {
                                "success": success,
                                "message": message,
                                "target_station": target,
                            }
                            
                        elif fname == "control_iot_device":
                            target = args.get("target_device_code")
                            action = args.get("action")

                            print(
                                f"✅ IoT: {target} için '{action}' KOMUTU ÇALIŞTIRILIYOR..."
                            )

                            # Komutu doğrudan çalıştır
                            success, message = self.execute_iot_command(target,action)

                            response_data = {
                                "success": success,
                                "message": message,
                                "target_device_code": target,
                                "action": action,
                            }
                        if response_data:
                            # Tool sonucunu modele geri besle
                            if fname != "sense_of_response":
                                tool_response = chat.send_message_stream(
                                    f"[SYSTEM] Tool {fname} executed. Result: {response_data}. Inform user."
                                )
                                # Tool cevabını da seslendir
                                tool_buffer = ""
                                for tr_chunk in tool_response:
                                    if tr_chunk.text:
                                        t_txt = tr_chunk.text
                                        tool_buffer += t_txt
                                        full_response_text += t_txt
                                        if re.search(r'[.!?:\n]', t_txt):
                                            self._stream_tts(tool_buffer)
                                            tool_buffer = ""
                                if tool_buffer.strip(): self._stream_tts(tool_buffer)
                        self.text_received.emit("System", f"(Sistem: {response_data})")
                        #self._stream_tts(res_txt)
                        full_response_text += f"[TOOL] {tool_response}"

            if buffer.strip(): self._stream_tts(buffer)

            if full_response_text.strip():
                # DÜZELTME 2: Geçmişi manuel güncellerken DOĞRU FORMATI kullanıyoruz.
                # String değil, {"text": "..."} objesi olmalı.
                
                # Kullanıcı mesajını geçmişe ekle
                self.chat_history.append({
                    "role": "user",
                    "parts": [{"text": augmented_user_text}]
                })

                # Model cevabını geçmişe ekle
                self.chat_history.append({
                    "role": "model",
                    "parts": [{"text": full_response_text}]
                })
                
                # Hafızaya kaydet (Arka planda)
                threading.Thread(
                    target=self.memory_manager.add_interaction, 
                    args=(user_text, full_response_text)
                ).start()

            self.status_changed.emit("Hazır")

            try: self.chat_history = chat.get_history()
            except: pass

            self.status_changed.emit("Hazır")
        except Exception as e:
            print(f"LLM Error: {e}")
            self.status_changed.emit("Zeka hatası.")

    def execute_iot_command(self, target_code: str, action: str):
        """Gerçek IoT eylemi."""
        try:
            if target_code in self.light_device_map:
                device_info = self.light_device_map[target_code]
                group = device_info["group"]
                index = device_info["index"]
                if action == "turn_on":
                    self.iot.send_data_for_light_func(
                        group, index, switch=True, dimming=150
                    )
                    print(f"*** SİMÜLASYON: {target_code} AÇILDI ***")
                    return True, f"{target_code} başarıyla açıldı."
                elif action == "turn_off":
                    self.iot.send_data_for_light_func(
                        group, index, switch=False, dimming=0
                    )
                    print(f"*** SİMÜLASYON: {target_code} KAPATILDI ***")
                    return True, f"{target_code} başarıyla kapatıldı."
            return False, f"Cihaz bulunamadı: {target_code}"
        except Exception as e:
            print(f"execute_iot_command Hata: {e}")
            return False, f"Hata: {e}"
        
    def execute_navigation_command(self, target_station: str):
        """Gerçek navigasyon isteğini ROS endpoint'ine gönderir."""
        if target_station not in self.station_names:
            print(f"*** HATA: Bilinmeyen istasyon: {target_station} ***")
            return False, f"Bilinmeyen istasyon: {target_station}"

        payload = {
            "station": target_station,
            "source": ROBOT_ID,
            "ts": int(time.time()),
        }

        try:
            print(
                f"*** NAVİGASYON: {self.ROS_NAV_ENDPOINT} adresine {payload} gönderiliyor... ***"
            )
            response = requests.post(self.ROS_NAV_ENDPOINT, json=payload, timeout=5)
            response.raise_for_status()
            print(f"*** NAVİGASYON BAŞLATILDI: {target_station} ***")
            return True, f"Navigasyon {target_station} hedefine başarıyla başlatıldı."
        except requests.exceptions.RequestException as e:
            print(f"execute_navigation_command Hata: {e}")
            return False, f"Navigasyon servisine bağlanılamadı: {e}"

    def _stream_tts(self, text):
        try:
            if not text or len(text.strip()) < 2: return
            audio_stream = self.eleven_client.text_to_speech.stream(
                text=text, voice_id="pNInz6obpgDQGcFmaJgB", model_id="eleven_multilingual_v2", output_format="pcm_22050"
            )
            for chunk in audio_stream: self.audio_out_queue.put(chunk)
        except: pass

    def cleanup(self):
        self.running = False
# ============================================================================
# 4. ENTEGRE GUI
# ============================================================================
class IntegratedGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beezy AI - Vision & Voice Integrated")
        self.resize(1200, 800)
        
        # --- Workerlar ---
        # 1. AI Logic Worker
        self.ai_thread = QThread()
        self.ai_worker = AIWorker()
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_thread.start()

        # 2. Vision Worker (Kamera & Pan-Tilt)
        self.vision_worker = VisionWorker()
        
        self.setupUI()
        self.connect_signals()
        
        # Vision Başlat
        self.vision_worker.start()

    def setupUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- SOL PANEL: KAMERA VE KONTROL ---
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        # Kamera Görüntüsü
        self.video_label = QLabel("Kamera Başlatılıyor...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)

        # Kullanıcı Bilgisi
        self.user_info_label = QLabel("Tanınan Kişi: Unknown")
        self.user_info_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2E7D32;")
        self.user_info_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.user_info_label)

        # Beni Hatırla Butonu
        self.btn_remember = QPushButton("📸 Beni Hatırla (Yeni Kayıt)")
        self.btn_remember.setFixedHeight(50)
        self.btn_remember.setStyleSheet("background-color: #1976D2; color: white; font-size: 16px; border-radius: 8px;")
        self.btn_remember.clicked.connect(self.handle_remember_me)
        left_layout.addWidget(self.btn_remember)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # --- SAĞ PANEL: CHAT VE ASİSTAN ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        self.chat_area = QTextBrowser()
        self.chat_area.setStyleSheet("font-size: 14px; padding: 10px;")
        right_layout.addWidget(self.chat_area)

        # Mikrofon Butonu
        self.mic_btn = QPushButton("🎤 Bas Konuş")
        self.mic_btn.setFixedSize(120, 120)
        self.mic_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; border-radius: 60px; font-size: 16px; font-weight: bold;
            }
            QPushButton:pressed { background-color: #388E3C; }
        """)
        self.mic_btn.clicked.connect(self.toggle_mic)
        right_layout.addWidget(self.mic_btn, 0, Qt.AlignCenter)
        
        self.status_label = QLabel("Hazır")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        main_layout.addWidget(right_panel)

    def connect_signals(self):
        # Vision Sinyalleri
        self.vision_worker.change_pixmap_signal.connect(self.update_image)
        self.vision_worker.face_recognized_signal.connect(self.handle_face_recognition)
        self.vision_worker.system_log_signal.connect(lambda msg: self.chat_area.append(f"<i style='color:gray'>{msg}</i>"))

        # AI Sinyalleri
        self.ai_worker.text_received.connect(self.append_chat)
        self.ai_worker.status_changed.connect(self.status_label.setText)

    @Slot(QImage)
    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    @Slot(str)
    def handle_face_recognition(self, user_id):
        # Sadece değişim olduğunda güncelle
        current_text = self.user_info_label.text()
        if user_id not in current_text:
            self.user_info_label.setText(f"Tanınan Kişi: {user_id}")
            # AI Worker'ın hafızasını güncelle
            self.ai_worker.update_active_user(user_id)

    def handle_remember_me(self):
        text, ok = QInputDialog.getText(self, 'Beni Hatırla', 'Adınız nedir?')
        if ok and text:
            # Türkçe karakterleri ve boşlukları temizle
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', text)
            if not clean_name: clean_name = f"User_{int(time.time())}"
            
            # Vision Worker'a emir ver
            self.vision_worker.set_register_mode(clean_name)
            QMessageBox.information(self, "Hazır", f"Lütfen kameraya bakın. '{clean_name}' olarak kaydedilecek.")

    def toggle_mic(self):
        if self.ai_worker.is_ai_speaking:
            self.ai_worker.interrupt_playback()
            self.mic_btn.setText("🔴 Dinliyor")
            self.mic_btn.setStyleSheet("background-color: #F44336; color: white; border-radius: 60px;")
            self.ai_worker.start_recording_session()
        elif not self.ai_worker.is_recording:
            self.mic_btn.setText("🔴 Bırak")
            self.mic_btn.setStyleSheet("background-color: #F44336; color: white; border-radius: 60px;")
            self.ai_worker.start_recording_session()
        else:
            self.mic_btn.setText("🎤 Bas Konuş")
            self.mic_btn.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 60px;")
            self.ai_worker.stop_recording_session()

    @Slot(str, str)
    def append_chat(self, sender, message):
        color = "blue" if sender == "User" else "green" if sender == "AI" else "black"
        self.chat_area.append(f"<b style='color:{color}'>{sender}:</b> {message}")

    def closeEvent(self, event):
        self.vision_worker.stop()
        self.ai_worker.cleanup()
        self.ai_thread.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntegratedGUI()
    window.show()
    sys.exit(app.exec())