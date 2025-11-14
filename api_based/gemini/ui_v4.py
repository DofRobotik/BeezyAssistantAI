import sys
import asyncio
import threading
import os
import time
import json
import traceback
import queue  # Sadece asyncio thread-safe olmayan GUI iletişimi için
from typing import Optional, Tuple
import re
from urllib.parse import urlparse, parse_qs

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QMessageBox,
    QGraphicsDropShadowEffect,
    QTextBrowser,
)
from PyQt6.QtCore import (
    Qt,
    QTimer,
    pyqtSignal as Signal,
    pyqtSlot as Slot,
    QThread,
    QObject,
    QUrl,
)
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile

# Gerekli modüllerin import edilmesi
import pyaudio
from google import genai
from google.genai import types
from googleapiclient.discovery import build
from dotenv import load_dotenv
import requests

# IoT fonksiyonelliği
from iot import AmrLoungeClass

# Environment değişkenlerini yükle
load_dotenv()


class GeminiLiveWorker(QObject):
    """
    Tüm Gemini Live API entegrasyonunu ve ses işlemlerini
    'asyncio' tabanlı bir streaming mimarisiyle yöneten worker.
    Bu, 'v3.py' script'indeki mantığı temel alır.
    """

    # Ana thread (GUI) ile iletişim için sinyaller
    status_changed = Signal(str)
    response_received = Signal(str)
    link_received = Signal(str, str)  # <-- YENİ SİNYAL (url, title)
    error_occurred = Signal(str)
    turn_finished = Signal()

    def __init__(self):
        super().__init__()
        self.loop = None
        self.session = None
        self.is_recording = False  # GUI butonu tarafından kontrol edilir
        self._playback_muted = False  # Barge-in için
        self.main_async_task = None  # Ana asyncio görevini tutar

        # Ses konfigürasyonu
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.SEND_SAMPLE_RATE = 16000
        self.RECEIVE_SAMPLE_RATE = 24000
        self.CHUNK_SIZE = 1024

        # Asyncio kuyrukları (v3.py'den)
        self.audio_in_queue = None  # Modelden gelen ses (oynatmak için)
        self.audio_out_queue = None  # Mikrofondan giden ses (modele göndermek için)

        # API Key
        self.GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        if not self.GOOGLE_API_KEY:
            self.error_occurred.emit(
                "GOOGLE_API_KEY bulunamadı. .env dosyasını kontrol edin."
            )
            return

        # Model konfigürasyonu
        self.MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

        # PyAudio
        self.pya = pyaudio.PyAudio()
        self._latest_metadata = None

        # Gemini Client
        try:
            self.client = genai.Client(
                http_options={"api_version": "v1alpha"}, api_key=self.GOOGLE_API_KEY
            )
            self.setup_tools_and_config()  # <-- Bu fonksiyonu GÜNCELLEDİK
            self.status_changed.emit("Asistan başlatıldı.")
        except Exception as e:
            self.error_occurred.emit(f"Gemini istemcisi başlatılamadı: {str(e)}")

    def setup_tools_and_config(self):
        """Gemini için tool'ları ve config'i hazırlar (v3.py'den)"""
        try:
            # IoT Cihazları
            iot_service_url = "10.10.10.244"
            iot_port = 3001
            self.iot = AmrLoungeClass(iot_service_url, iot_port)

            self.light_device_map = {}
            self.all_iot_device_codes = []
            for group_index, devices in self.iot._AmrLoungeClass__lounge_place.items():
                for place_index, device in enumerate(devices):
                    code = device["code"]
                    self.all_iot_device_codes.append(code)
                    self.light_device_map[code] = {
                        "group": group_index,
                        "index": place_index,
                    }
            iot_device_prompt_list = " ,".join(self.all_iot_device_codes)

        except Exception as e:
            self.error_occurred.emit(f"IoT kurulum hatası: {str(e)}")
            self.all_iot_device_codes = []
            self.light_device_map = {}
            iot_device_prompt_list = ""

        # Navigasyon (v3.py'den)
        self.ROS_NAV_ENDPOINT = "http://10.10.190.14:8000/navigate"
        self.ROBOT_ID = "amr-1"
        self.stations = [
            {
                "name": "station_a",
                "property": "Food Court, a great place for drink and eat. Related to food, hunger, restaurant.",
            },
            {
                "name": "station_b",
                "property": "Restrooms area. Related to WC, toilet, bathroom, washroom, pee, urinate, relief.",
            },
            {
                "name": "station_c",
                "property": "Fun room area. Related to play, games, entertainment, fun, joy, relax, amusement, leisure.",
            },
            {
                "name": "station_d",
                "property": "A garment shop. Related to clothes, fashion, dressing.",
            },
            {
                "name": "station_e",
                "property": "A tech shop. Related to technology, electronics, phone, computer.",
            },
        ]
        self.station_names = [s["name"] for s in self.stations]
        self.station_prompt_list = "\n".join(
            [f"- {s['name']}: {s['property']}" for s in self.stations]
        )
        self.emotions = ["happy", "sad"]

        # Tools (v3.py'den) - YENİ TOOL EKLENDİ
        tools = [
            types.Tool(
                function_declarations=[
                    # 1. IoT Tool
                    types.FunctionDeclaration(
                        name="control_iot_device",
                        description="Turns on/off IoT devices. Always asks for confirmation.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "target_device_code": {
                                    "type": "string",
                                    "enum": self.all_iot_device_codes,
                                },
                                "action": {
                                    "type": "string",
                                    "enum": ["turn_on", "turn_off"],
                                },
                                "reason": {"type": "string"},
                            },
                            "required": [
                                "target_device_code",
                                "action",
                                "reason",
                            ],
                        },
                    ),
                    # 2. Navigasyon Tool
                    types.FunctionDeclaration(
                        name="navigate_to_station",
                        description="Guides the robot to a specific station. Always asks for confirmation.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "target_station": {
                                    "type": "string",
                                    "enum": self.station_names,
                                },
                                "reason": {"type": "string"},
                            },
                            "required": ["target_station", "reason"],
                        },
                    ),
                    # 3. Emotion Tool
                    types.FunctionDeclaration(
                        name="sense_of_response",
                        description="Sense of Assistant's response. Will directly used to show user response emotion by LED panels.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "emotion": {"type": "string", "enum": self.emotions},
                            },
                            "required": ["emotion"],
                        },
                    ),
                    types.FunctionDeclaration(
                        name="find_youtube_video",
                        description="Searches YouTube for a specific video query and prepares it for display.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "search_query": {
                                    "type": "string",
                                    "description": "The search query, e.g., 'pancake recipe' or 'latest tech news'",
                                },
                            },
                            "required": ["search_query"],
                        },
                    ),
                ]
            ),
            types.Tool(google_search=types.GoogleSearch()),
        ]

        # --- GÜNCELLENMİŞ SİSTEM PROMPT'U (İNGİLİZCE) ---
        system_instruction_prompt = (
            "You are Beezy, a helpful, friendly, and proactive service robot assistant from DOF Robotics. "
            "Your **permanent location** is the Cevahir AVM in Türkiye. You are never lost and you always know you are in this mall.\n\n"
            "Your **primary goal** is to assist visitors. Your main capabilities are:\n"
            "1.  **Navigation:** Guiding users to specific stations within the mall.\n"
            "2.  **IoT Control:** Controlling prototype devices (lights).\n"
            "3.  **General Conversation:** Answering questions (using Google Search).\n\n"
            "## CORE BEHAVIOR: BE PROACTIVE WITH NAVIGATION ##\n"
            "This is your most important rule. You are a mobile robot.\n"
            f"You have a defined list of navigation stations:\n{self.station_prompt_list}\n"
            "When a user asks about a location, a need, or an activity (e.g., 'I'm hungry', 'Where can I eat?', 'I need a restroom'), "
            "you **MUST** check if one of your stations matches that need.\n"
            "If a match is found, your **first response** must be to **offer navigation**.\n\n"
            "**Example Interaction:**\n"
            "  * **User:** 'Buralarda yemek yiyebileceğim bir yer var mı?'\n"
            "  * **WRONG Response:** 'Üzgünüm, nerede olduğunuzu bilmiyorum.' (This is wrong. You ALWAYS know you are in Cevahir AVM).\n"
            "  * **WRONG Response:** 'Food Court'ta yemek yiyebilirsiniz.' (This is not helpful, you are a robot, you must offer to GUIDE them).\n"
            "  * **CORRECT Response:** 'Elbette, 'station_a' (Food Court) alanımız var. Sizi oraya götürmemi ister misiniz?' (You will then wait for their answer).\n\n"
            "## TOOL USAGE RULES (CRITICAL) ##\n\n"
            "**1. Navigation (navigate_to_station) & IoT (control_iot_device) - CONVERSATIONAL CONFIRMATION**\n\n"
            "This is a **CRITICAL TWO-TURN** process that *you* must manage conversationally. The system will NOT send any 'needs_confirmation' flags. You must handle it.\n\n"
            "**TURN 1: THE OFFER**\n"
            "   * When a user's request requires navigation or IoT control, you **MUST NOT** call the tool.\n"
            "   * Your *only* action in this turn is to **verbally ask the user for confirmation**.\n"
            "   * **Example (Nav):** 'I can guide you to Station A (Food Court). Would you like that?'\n"
            "   * **Example (IoT):** 'You want the lounge light on? Should I turn on LOUNGE_GENEL?'\n"
            "   * After asking, you will **STOP** and wait for the user's response (e.g., 'Yes', 'No', 'Evet', 'Hayır').\n\n"
            "**TURN 2: THE EXECUTION (If user confirms)**\n"
            "   * If (and only if) the user confirms your offer in their response (e.g., 'Yes', 'Evet').\n"
            "   * In your *next* response turn, you **MUST** do **TWO** things:\n"
            "       1.  **Verbally acknowledge** the action (e.g., 'Okay, starting navigation to the food court.' or 'Alright, turning on the lounge light.')\n"
            "       2.  **AND** in the *same turn*, call the corresponding tool (`navigate_to_station` or `control_iot_device`) with the correct arguments.\n\n"
            "**Example (Nav) Flow:**\n"
            "   * **User:** 'I'm hungry.'\n"
            "   * **Beezy (Turn 1):** 'We have a Food Court at station_a. Shall I guide you there?' [NO TOOL CALL]\n"
            "   * **User:** 'Yes, please.'\n"
            "   * **Beezy (Turn 2):** 'Okay, starting navigation to station_a.' [CALLS: `navigate_to_station(target_station='station_a')`]\n\n"
            "**Example (IoT) Flow:**\n"
            "   * **User:** 'It's dark in here, can you turn on the main light?'\n"
            "   * **Beezy (Turn 1):** 'Sure, I can turn on the LOUNGE_GENEL light. Is that okay?' [NO TOOL CALL]\n"
            "   * **User:** 'Evet, lütfen.'\n"
            "   * **Beezy (Turn 2):** 'Tamamdır, LOUNGE_GENEL ışığını açıyorum.' [CALLS: `control_iot_device(target_device_code='LOUNGE_GENEL', action='turn_on', reason='user request')`]\n\n"
            "**2. Emotion Sensing (sense_of_response):**\n"
            "   * Call this tool to match your **own** verbal response's emotion.\n"
            "   * Example: If you say 'I'm sorry, I can't help with that', you must also call `sense_of_response(emotion='sad')`.\n"
            "   * Example: If you say 'Certainly! I can take you there!', you must also call `sense_of_response(emotion='happy')`.\n\n"
            "**3. Using Search Tools (CRITICAL)**\n"
            "   You have TWO different search tools. You must choose the correct one based on the user's *intent*.\n\n"
            "   **A. For General Questions (Using `GoogleSearch`)**\n"
            "      * **USE FOR:** Factual questions, weather, definitions, 'who is', 'what is' (e.g., 'Who is the president?', 'What's the weather?', 'Bana X hakkında bilgi ver').\n"
            "      * **INTENT:** The user wants a *verbal answer* from you.\n"
            "      * **CRITICAL: DO NOT** use this tool if the user's request includes the word 'video', 'YouTube', 'show', 'izle', or 'göster'. For those, use `find_youtube_video`.\n\n"
            "   **B. For Showing Videos (Using `find_youtube_video`)**\n"
            "      * **USE FOR:** *Any* request where the user's intent is to *watch a video*.\n"
            "      * **INTENT:** The user wants to *see* content on the screen.\n"
            "      * **TRIGGER WORDS:** 'video', 'YouTube', 'show me', 'izlet', 'göster', 'watch', 'bul', 'arIyorum' (e.g., 'Show me a pancake video', 'Bana komik kedi videoları bul', 'Kuantum fiziği hakkında video arıyorum').\n"
            "      * **RULE:** If the request contains the word 'video', 'YouTube', or 'izle', you **MUST** prefer this tool over `GoogleSearch`, even if they also use words like 'search' or 'arIyorum'.\n\n"
            "   * **Example (The exact problem we are fixing):**\n"
            "     * **User:** 'Kuantum fiziği hakkında bir video arıyorum.'\n"
            "     * **WRONG:** (Calling `GoogleSearch` because of 'arIyorum')\n"
            "     * **CORRECT (Turn):** 'Elbette, kuantum fiziği hakkında bir video arıyorum.' [CALLS: `find_youtube_video(search_query='kuantum fiziği')`]\n\n"
            "**4. Language:**\n"
            "   * You **MUST** respond in the same language the user is speaking (e.g., Turkish or English).\n"
            ")"
            # (Değişkenin sonu)
        )

        self.CONFIG = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system_instruction_prompt,
            tools=tools,
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                )
            ),
            proactivity=types.ProactivityConfig(proactive_audio=True),
            context_window_compression=(
                types.ContextWindowCompressionConfig(
                    sliding_window=types.SlidingWindow()
                )
            ),
            thinking_config=types.ThinkingConfig(thinking_budget=512),
        )

    # --- IoT ve Navigasyon Yürütme Fonksiyonları (v3.py'den) ---
    def search_youtube_video(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Google Resmi Kütüphanesi ile YouTube araması yapar.
        """
        if not self.GOOGLE_API_KEY:
            print("HATA: API Key yok.")
            return None

        try:
            # Resmi istemciyi API Key ile başlatıyoruz (Login gerekmez)
            youtube = build("youtube", "v3", developerKey=self.GOOGLE_API_KEY)

            request = youtube.search().list(
                part="snippet",
                q=query,
                maxResults=1,
                type="video",
                videoEmbeddable="true",
            )

            # İsteği çalıştır
            response = request.execute()

            if "items" in response and len(response["items"]) > 0:
                item = response["items"][0]
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]

                embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0&iv_load_policy=3"
                return embed_url, title
            else:
                print(f"Sonuç bulunamadı: {query}")

        except Exception as e:
            # Detaylı hata mesajını görelim
            print(f"YouTube API Hatası (Resmi Lib): {e}")

        return None

    def execute_iot_command(self, target_code: str, action: str) -> Tuple[bool, str]:
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

    def execute_navigation_command(self, target_station: str) -> Tuple[bool, str]:
        """Gerçek navigasyon isteğini ROS endpoint'ine gönderir."""
        if target_station not in self.station_names:
            print(f"*** HATA: Bilinmeyen istasyon: {target_station} ***")
            return False, f"Bilinmeyen istasyon: {target_station}"

        payload = {
            "station": target_station,
            "source": self.ROBOT_ID,
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

    # --- Async Çekirdek (v3.py'den) ---

    @Slot()
    def run_async_loop(self):
        """QThread başladığında bu fonksiyon çalışır, asyncio loop'u kurar."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.main_async_task = self.loop.create_task(self._async_run())
            self.loop.run_forever()
        except Exception as e:
            self.error_occurred.emit(f"Async loop hatası: {e}")
        finally:
            print("Asyncio loop kapatılıyor...")
            if self.loop.is_running():
                print("Uyarı: Loop run_forever'dan çıktı ama hala 'running' görünüyor.")

            self.loop.close()
            print("Asyncio loop kapatıldı.")
            self.pya.terminate()
            print("PyAudio sonlandırıldı.")

    # <-- GÜNCELLENDİ: YENİDEN BAĞLANMA MANTIĞI EKLENDİ ---
    async def _async_run(self):
        """Ana async fonksiyonu - Artık yeniden bağlanmayı deneyecek."""

        while True:  # <-- YENİ: Yeniden bağlanma döngüsü
            tasks = set()
            try:
                # Bağlantıyı kur
                async with self.client.aio.live.connect(
                    model=self.MODEL, config=self.CONFIG
                ) as session:
                    self.session = session
                    self.status_changed.emit("Bağlantı kuruldu. Dinlemeye hazır!")

                    self.audio_in_queue = asyncio.Queue()
                    self.audio_out_queue = asyncio.Queue(maxsize=100)

                    # Görevleri oluştur
                    tasks.add(asyncio.create_task(self._send_realtime()))
                    tasks.add(asyncio.create_task(self._listen_audio()))
                    tasks.add(asyncio.create_task(self._receive_audio()))
                    tasks.add(asyncio.create_task(self._play_audio()))

                    # Görevlerin bitmesini bekle
                    await asyncio.gather(*tasks)

            except (asyncio.CancelledError, KeyboardInterrupt):
                print("\nAsync run sonlandırılıyor (CancelledError)...")
                break  # İptal istendi, ana döngüden (while True) çık

            except Exception as e:
                # Bu blok, _receive_audio'dan fırlatılan 1011 hatasını yakalayacak
                print(f"Ana '_async_run' döngüsünde hata (yeniden denenecek): {e}")
                self.error_occurred.emit(
                    f"Bağlantı hatası: {e}. 5sn içinde yeniden denenecek..."
                )

                # Hata oluştuğunda tüm alt görevleri iptal et (önemli)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(5)  # Yeniden bağlanmadan önce 5 saniye bekle

            finally:
                # Bu 'finally' bloğu, 'async with' bloğundan
                # *her* çıkıldığında (hata veya normal) çalışır
                print("Async görevler (iç döngü) temizleniyor...")
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    # Görevlerin iptal işlemini bitirmesini bekle
                    await asyncio.gather(*tasks, return_exceptions=True)
                print("İç görevler temizlendi.")

        # Bu noktaya sadece CancelledError veya KeyboardInterrupt ile gelinmeli
        print("Ana yeniden bağlanma döngüsü (while True) sonlandı.")
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

    # --- GÜNCELLENEN _async_run SONU ---

    async def _send_realtime(self):
        """Kuyruktaki ses verisini Gemini'a gönderir (v3.py'den)"""
        while True:
            try:
                msg = await self.audio_out_queue.get()
                blob = types.Blob(
                    data=msg["data"],
                    mime_type=msg.get("mime_type", "audio/pcm;rate=16000"),
                )
                await self.session.send_realtime_input(audio=blob)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"_send_realtime hatası: {e}")

    async def _listen_audio(self):
        """Mikrofonu dinler ve kuyruğa atar (v3.py'den)"""
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        mic_info = self.pya.get_default_input_device_info()

        while True:
            if not self.is_recording:
                await asyncio.sleep(0.01)
                continue

            stream = None
            try:
                if not self.is_recording:
                    continue
                stream = await asyncio.to_thread(
                    self.pya.open,
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.SEND_SAMPLE_RATE,
                    input=True,
                    input_device_index=mic_info["index"],
                    frames_per_buffer=self.CHUNK_SIZE,
                )
                print("Stream açıldı, dinleniyor...")

                while self.is_recording:
                    try:
                        data = await asyncio.to_thread(
                            stream.read, self.CHUNK_SIZE, **kwargs
                        )
                        await self.audio_out_queue.put(
                            {"data": data, "mime_type": "audio/pcm"}
                        )
                    except IOError as e:
                        if getattr(e, "errno", None) == pyaudio.paInputOverflowed:
                            continue
                        break
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Bilinmeyen _listen_audio hatası: {e}")
            finally:
                if stream:
                    await asyncio.to_thread(stream.stop_stream)
                    await asyncio.to_thread(stream.close)
                    print("Stream kapatıldı.")

    async def _receive_audio(self):
        """Modelden gelen yanıtları (ses, metin, tool) işler (YENİLENMİŞ VERSİYON)"""
        while True:
            # Her 'turn' için gönderilen URL'leri takip et (duplikasyonu önler)
            sent_urls_this_turn = set()
            link_to_send_at_end = None
            try:
                turn = self.session.receive()

                # --- 1. Adım: Tüm chunk'ları (ses, metin, tool) işle ---
                async for chunk in turn:

                    # 1. Sunucu İçeriği (Ses, Metin VE METADATA)
                    if chunk.server_content:

                        # --- Ses ve metin verisini işle (Bu kısım aynı kaldı) ---
                        if data := chunk.data:
                            self.audio_in_queue.put_nowait(data)

                        if text := chunk.text:
                            print(f"AI: {text}", end="")
                            # GUI'yi metin hakkında bilgilendir
                            self.response_received.emit(f"📝 AI: {text}")

                    # 2. Araç Çağrısı (Function Call) - SADELEŞTİRİLMİŞ MANTIK
                    elif chunk.tool_call:
                        print(f"\n[🔄 Araç Çağrısı Algılandı]")
                        self.response_received.emit(f"[🔄 Araç Çağrısı Algılandı...]")
                        function_responses_to_send = []

                        for fc in chunk.tool_call.function_calls:
                            try:
                                args = fc.args
                                response_data = {
                                    "success": False,
                                    "message": "Bilinmeyen fonksiyon",
                                }

                                # --- 'sense_of_response' KONTROLÜ (Aynı kalıyor) ---
                                if fc.name == "sense_of_response":
                                    emotion = args.get("emotion")
                                    if emotion:
                                        print(f"--- 🤖 MODEL DUYGUSU: {emotion} ---")
                                        self.response_received.emit(
                                            f"🤖 Duygu: {emotion}"
                                        )
                                        response_data = {
                                            "success": True,
                                            "message": f"Sonuç duygu:{emotion}",
                                        }

                                elif fc.name == "find_youtube_video":
                                    query = args.get("search_query")
                                    if not query:
                                        raise ValueError(
                                            "Arama sorgusu (search_query) eksik."
                                        )

                                    print(
                                        f"--- 🔎 YouTube API Araması (Tool Call) Başlatılıyor: '{query}' ---"
                                    )
                                    self.response_received.emit(
                                        f"🔎 YouTube'da aranıyor: {query}"
                                    )

                                    # Bizim özel YouTube API fonksiyonumuzu çağır
                                    result = await asyncio.to_thread(
                                        self.search_youtube_video, query
                                    )

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
                                        print(f"--- ❌ VİDEO BULUNAMADI: {query} ---")
                                        self.response_received.emit(
                                            f"❌ Video bulunamadı: {query}"
                                        )
                                        response_data = {
                                            "success": False,
                                            "message": f"Video bulunamadı: {query}",
                                        }

                                # --- YENİ EYLEM MANTIĞI (Onay mantığı kaldırıldı) ---
                                # Model bu fonksiyonları çağırdığında, onayın zaten
                                # sözlü olarak alındığını varsayıyoruz.

                                # Durum 1: IoT (control_iot_device)
                                elif fc.name == "control_iot_device":
                                    target = args.get("target_device_code")
                                    action = args.get("action")

                                    print(
                                        f"✅ IoT: {target} için '{action}' KOMUTU ÇALIŞTIRILIYOR..."
                                    )
                                    self.response_received.emit(
                                        f"✅ IoT: {target} için '{action}' komutu çalıştırılıyor..."
                                    )

                                    # Komutu doğrudan çalıştır
                                    success, message = await asyncio.to_thread(
                                        self.execute_iot_command, target, action
                                    )

                                    response_data = {
                                        "success": success,
                                        "message": message,
                                        "target_device_code": target,
                                        "action": action,
                                    }

                                # Durum 2: Navigasyon (navigate_to_station)
                                elif fc.name == "navigate_to_station":
                                    target = args.get("target_station")

                                    print(
                                        f"✅ Navigasyon: {target} hedefine yönlendiriliyor..."
                                    )
                                    self.response_received.emit(
                                        f"✅ Navigasyon: {target} hedefine yönlendiriliyor..."
                                    )

                                    # Komutu doğrudan çalıştır
                                    success, message = await asyncio.to_thread(
                                        self.execute_navigation_command, target
                                    )

                                    response_data = {
                                        "success": success,
                                        "message": message,
                                        "target_station": target,
                                    }

                                # --- Yürütme Bitti ---
                                self.response_received.emit(
                                    f"✅ Sonuç: {response_data['message']}"
                                )
                                function_responses_to_send.append(
                                    types.FunctionResponse(
                                        id=fc.id, name=fc.name, response=response_data
                                    )
                                )

                            except Exception as e:
                                # (Hata yönetimi aynı kalıyor)
                                print(f"❌ Fonksiyon işleme hatası: {e}")
                                self.error_occurred.emit(f"Fonksiyon hatası: {e}")
                                function_responses_to_send.append(
                                    types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"success": False, "message": str(e)},
                                    )
                                )

                        # (Fonksiyon yanıtı gönderme kısmı aynı kalıyor)
                        if function_responses_to_send:
                            print(
                                f"[📬 {len(function_responses_to_send)} adet fonksiyon yanıtı gönderiliyor...]"
                            )
                            await self.session.send_tool_response(
                                function_responses=function_responses_to_send
                            )

                # --- Turn'ün bittiği yer ---

                print("Turn tamamlandı.")

                # (Barge-in ve 'turn_finished' sinyal mantığı aynı kalıyor)
                if link_to_send_at_end:
                    url, title = link_to_send_at_end
                    print(f"--- 📺 Gecikmeli Video Sinyali Gönderiliyor: {title} ---")
                    self.link_received.emit(url, title)
                    # Sadece bir kez gönder, bir sonraki turn'e taşıma
                    link_to_send_at_end = None

                if self.is_recording:
                    print("Barge-in algılandı: 'turn_finished' sinyali gönderilmedi.")
                    continue

                print("Turn normal bitti: 'turn_finished' sinyali gönderiliyor.")
                self.turn_finished.emit()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Hata: '_receive_audio' akışında sorun: {e}")
                self.error_occurred.emit(f"Yanıt alma hatası: {e}")
                # Hatayı yeniden fırlat (yeniden bağlanma döngüsü için)
                raise e

    async def _interrupt_playback(self):
        """Mevcut ses oynatmayı anında keser (barge-in) - GÜVENLİ VERSİYON"""
        print("Barge-in: Oynatma kesiliyor (mute + clear)...")

        # 1. Gelecekteki oynatmaları durdur
        self._playback_muted = True

        # 2. Kuyruktaki bekleyen sesleri temizle
        await self._clear_audio_queue_async()

        # 3. Stream'i kapatmıyoruz. _play_audio'daki bayrak yeterli.
        print("Barge-in: Mute edildi ve kuyruk temizlendi.")

    async def _play_audio(self):
        """Gelen sesi oynatır (Basit ve sağlam versiyon)"""
        stream = None
        try:
            stream = await asyncio.to_thread(
                self.pya.open,
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RECEIVE_SAMPLE_RATE,
                output=True,
            )
            print("Ses oynatıcı (basit) hazır.")
            while True:
                bytestream = await self.audio_in_queue.get()

                if self._playback_muted or self.is_recording:
                    self.audio_in_queue.task_done()
                    continue

                await asyncio.to_thread(stream.write, bytestream)
                self.audio_in_queue.task_done()

        except asyncio.CancelledError:
            print("Ses oynatıcı (basit) iptal edildi.")
        except Exception as e:
            print(f"Ses oynatıcı (basit) hatası: {e}")
            self.error_occurred.emit(f"Ses oynatma hatası: {e}")
        finally:
            if stream:
                await asyncio.to_thread(stream.stop_stream)
                await asyncio.to_thread(stream.close)
            print("Ses oynatıcı (basit) kapatıldı.")

    async def _clear_audio_queue_async(self):
        """Async olarak gelen ses kuyruğunu temizler (Barge-in için)"""
        if self.audio_in_queue is None:
            return
        try:
            while True:
                self.audio_in_queue.get_nowait()
                self.audio_in_queue.task_done()
        except asyncio.QueueEmpty:
            pass

    # --- GUI Tarafından Çağrılan Slotlar ---

    @Slot()
    def start_recording(self):
        """GUI 'Başlat' butonuna bastığında tetiklenir."""
        if self.is_recording or not self.session or not self.loop:
            return

        print("🔴 Kayıt başlıyor (GUI)...")
        self.is_recording = True

        asyncio.run_coroutine_threadsafe(self._interrupt_playback(), self.loop)

        coro = self.session.send_realtime_input(activity_start=types.ActivityStart())
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    @Slot()
    def stop_processing(self):
        """GUI 'Durdur' butonuna bastığında tetiklenir."""
        if not self.is_recording or not self.session or not self.loop:
            return

        print("⚪ Kayıt durdu (GUI). İşleniyor...")
        self.is_recording = False
        self._playback_muted = False

        coro = self.session.send_realtime_input(activity_end=types.ActivityEnd())
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    @Slot()
    def stop(self):
        """Uygulama kapandığında ana async göreve iptal sinyali gönderir."""
        print("Worker stop çağrıldı.")
        if self.main_async_task and self.loop and self.loop.is_running():
            try:
                self.loop.call_soon_threadsafe(self.main_async_task.cancel)
            except RuntimeError as e:
                print(
                    f"Görev iptal edilirken hata (muhtemelen loop zaten kapanmış): {e}"
                )
            except Exception as e:
                print(f"Görev iptal edilirken bilinmeyen hata: {e}")


# --- PySide6 GUI Sınıfları (enhanced.py'den) ---


class AnimatedMicButton(QPushButton):
    """Özel animasyonlu mikrofon butonu"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 140)
        self.setObjectName("micButton")
        self.is_listening = False
        self.breath_timer = QTimer(self)
        self.breath_timer.timeout.connect(self.update_breath)
        self.breath_value = 0
        self.breath_direction = 1
        self.stop_listening_animation()

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 8)
        self.setGraphicsEffect(shadow)

    def start_listening_animation(self):
        self.is_listening = True
        self.breath_timer.start(80)
        self.setStyleSheet(
            """
            QPushButton#micButton {
                border: 4px solid #F44336; border-radius: 70px;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, stop:0 #EF5350, stop:1 #F44336);
                color: white; font-size: 16px; font-weight: bold;
            }"""
        )

    def stop_listening_animation(self):
        self.is_listening = False
        self.breath_timer.stop()
        self.setFixedSize(140, 140)
        self.setStyleSheet(
            """
            QPushButton#micButton {
                border: 4px solid #4CAF50; border-radius: 70px;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, stop:0 #81C784, stop:1 #4CAF50);
                color: white; font-size: 16px; font-weight: bold;
            }
            QPushButton#micButton:hover {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, stop:0 #A5D6A7, stop:1 #66BB6A);
            }"""
        )

    def update_breath(self):
        if not self.is_listening:
            return
        self.breath_value += self.breath_direction * 3
        if self.breath_value >= 20:
            self.breath_direction = -1
        elif self.breath_value <= 0:
            self.breath_direction = 1
        new_size = 140 + int(self.breath_value * 0.2)
        self.setFixedSize(new_size, new_size)


class EnhancedVoiceAssistantGUI(QMainWindow):
    """Ana uygulama penceresi"""

    def __init__(self):
        super().__init__()
        self.is_listening = False
        self.worker_thread = None
        self.worker = None
        self.setupUI()
        self.setup_worker()

    def setupUI(self):
        self.setWindowTitle("🎤 Beezy Assistant AI - v2 (Streaming)")
        # <-- GÜNCELLENDİ: Arka plan rengi (veya degrade)
        self.setStyleSheet(
            "QMainWindow { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F8F9FA, stop:1 #E9ECEF); }"
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)  # Spacing'i 25'e geri aldım, 20 de kalabilir

        screen = QApplication.primaryScreen().geometry()
        screen_height = screen.height()

        self.square_size = int(screen_height * 0.50)

        # --- BÖLÜM 1: BİLEŞENLERİ TANIMLAMA ---

        title_label = QLabel("🎤 Beezy Assistant AI")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            "QLabel { color: #2E7D32; font-size: 32px; font-weight: bold; padding: 25px; margin-bottom: 10px; }"
        )

        self.status_label = QLabel("Başlatılıyor...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "QLabel { color: #666; font-size: 18px; padding: 15px; background-color: white; border-radius: 12px; border: 2px solid #E0E0E0; margin: 10px; }"
        )
        # Genişliği sabitle (Önemli)
        self.status_label.setFixedWidth(self.square_size)

        mic_container = QWidget()
        mic_layout = QHBoxLayout(mic_container)
        mic_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mic_layout.setContentsMargins(0, 20, 0, 20)
        self.mic_button = AnimatedMicButton()
        self.mic_button.setText("🎤 Bas Konuş")
        self.mic_button.clicked.connect(self.toggle_listening)
        self.mic_button.setEnabled(False)
        mic_layout.addWidget(self.mic_button)

        self.web_view_container = QWidget()
        self.web_view_container.setObjectName("webViewContainer")
        self.web_view_container.setStyleSheet(
            """
            QWidget#webViewContainer {
                border: 2px solid #007BFF; 
                border-radius: 12px; 
                background-color: white;
            }
            """
        )
        web_layout = QVBoxLayout(self.web_view_container)
        web_layout.setContentsMargins(5, 5, 5, 5)

        self.web_view = QWebEngineView()

        profile = self.web_view.page().profile()
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        profile.setHttpUserAgent(user_agent)

        self.web_view.load(QUrl("about:blank"))

        # --- GELİŞTİRİLMİŞ WEB ENGINE AYARLARI ---
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False
        )
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.AllowWindowActivationFromJavaScript, True
        )
        # --- AYARLAR SONU ---

        # Web sayfası yükleme durumunu izle
        self.web_view.loadFinished.connect(self.on_web_page_loaded)
        self.web_view.loadStarted.connect(lambda: print("Web sayfası yükleniyor..."))

        web_layout.addWidget(self.web_view)

        self.web_view_container.setFixedSize(self.square_size, self.square_size)
        self.web_view_container.setVisible(False)

        self.response_area = QTextBrowser()
        self.response_area.setReadOnly(True)
        self.response_area.setPlaceholderText(
            "🔊 Sesli yanıtlar hoparlörden oynatılacak...\n\n📝 Aktivite logu burada görünecek..."
        )
        self.response_area.setStyleSheet(
            "QTextBrowser { border: 2px solid #E0E0E0; border-radius: 12px; padding: 20px; font-size: 15px; background-color: white; }"
        )
        self.response_area.setFixedWidth(self.square_size)
        self.response_area.setMinimumHeight(150)
        self.response_area.setMaximumHeight(200)

        instructions = QLabel(
            "🎙️ Mikrofona tıkla → Konuş → Tekrar tıkla → 🔊 Yanıtı dinle"
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet(
            "QLabel { color: #888; font-size: 14px; font-style: italic; padding: 15px; background-color: #F5F5F5; border-radius: 8px; border: 1px solid #DDD; }"
        )
        instructions.setFixedWidth(self.square_size)
        instructions.setWordWrap(True)

        # --- BÖLÜM 2: BİLEŞENLERİ LAYOUT'A EKLEME (SADECE 1 KEZ) ---

        # Üst boşluk
        main_layout.addStretch(1)

        # 1. Başlık (Merkezlenmiş)
        main_layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignCenter)

        # 2. Durum (Merkezlenmiş)
        main_layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignCenter)

        # 3. Web Tarayıcı (Yatayda ortalamak için HBox kullan)
        web_view_hbox = QHBoxLayout()
        web_view_hbox.addStretch(1)  # Sol boşluk
        web_view_hbox.addWidget(self.web_view_container)  # <--- SADECE BURADA EKLENİYOR
        web_view_hbox.addStretch(1)  # Sağ boşluk
        main_layout.addLayout(web_view_hbox)

        # 4. Mikrofon Butonu (Merkezlenmiş)
        main_layout.addWidget(
            mic_container, 0, Qt.AlignmentFlag.AlignCenter
        )  # <--- SADECE BURADA EKLENİYOR

        # 5. Log Alanı (Merkezlenmiş)
        main_layout.addWidget(self.response_area, 0, Qt.AlignmentFlag.AlignCenter)

        # 6. Talimatlar (Merkezlenmiş)
        main_layout.addWidget(instructions, 0, Qt.AlignmentFlag.AlignCenter)

        # Alt boşluk
        main_layout.addStretch(1)

    def setup_worker(self):
        """Worker thread'i kurar ve başlatır."""
        self.worker_thread = QThread()
        self.worker = GeminiLiveWorker()
        self.worker.moveToThread(self.worker_thread)

        # Sinyalleri bağla
        self.worker.status_changed.connect(self.update_status)
        self.worker.response_received.connect(self.add_response)
        self.worker.link_received.connect(self.handle_link)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.turn_finished.connect(self.on_turn_finished)

        self.worker_thread.started.connect(self.worker.run_async_loop)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

        QTimer.singleShot(1000, lambda: self.update_status("Asistan başlatılıyor..."))

    @Slot()
    def toggle_listening(self):
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        self.is_listening = True
        self.mic_button.start_listening_animation()
        self.mic_button.setText("🔴 Dinliyorum...")
        self.status_label.setStyleSheet(
            "QLabel { color: #F44336; font-size: 18px; font-weight: bold; padding: 15px; background-color: #FFEBEE; border-radius: 12px; border: 2px solid #F44336; margin: 10px; }"
        )
        self.web_view_container.setVisible(False)
        self.web_view.setUrl(QUrl("about:blank"))
        self.worker.start_recording()

    def stop_listening(self):
        """Stop listening and start processing"""
        self.is_listening = False
        self.mic_button.stop_listening_animation()
        self.mic_button.setText("🤔 İşleniyor...")
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: #FFA000; font-size: 18px; font-weight: bold; 
                padding: 15px; background-color: #FFF8E1; 
                border-radius: 12px; border: 2px solid #FFA000; margin: 10px;
            }
        """
        )
        self.worker.stop_processing()

    @Slot(str)
    def update_status(self, status: str):
        self.status_label.setText(status)

        if "Dinlemeye hazır" in status:
            self.status_label.setStyleSheet(
                "QLabel { color: #4CAF50; font-size: 18px; font-weight: bold; padding: 15px; background-color: #E8F5E8; border-radius: 12px; border: 2px solid #4CAF50; margin: 10px; }"
            )
            self.mic_button.setText("🎤 Bas Konuş")
            self.mic_button.setEnabled(True)
            self.mic_button.stop_listening_animation()
            self.is_listening = False

    @Slot(str)
    def add_response(self, response: str):
        """Aktivite loguna yanıtı ekler"""
        timestamp = time.strftime("%H:%M:%S")
        self.response_area.append(f"[{timestamp}] {response}")
        self.response_area.ensureCursorVisible()

    @Slot(str, str)
    def handle_link(self, url: str, title: str):
        """
        Worker'dan gelen linki açar - Clean embed view (video only)
        """
        self.add_response(f"🔗 Video bulundu: {title}")

        # Video ID'yi URL'den çıkar
        video_id = ""
        if "/embed/" in url:
            video_id = url.split("/embed/")[1].split("?")[0]
        elif "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        else:
            print(f"❌ Video ID çıkarılamadı: {url}")
            self.add_response(f"❌ Geçersiz YouTube URL formatı")
            return

        print(f"📹 Video ID: {video_id}")

        # Temiz embed görünümü - sadece video player
        # youtube-nocookie.com kullanarak daha iyi gizlilik ve daha az kısıtlama
        html_content = f"""
        <!DOCTYPE html>
        <html style="margin:0;padding:0;height:100%;width:100%;overflow:hidden;">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    border: 0;
                }}
                html, body {{
                    width: 100%;
                    height: 100%;
                    background: #000;
                    overflow: hidden;
                }}
                iframe {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                }}
            </style>
        </head>
        <body>
            <iframe 
                src="https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&controls=1&modestbranding=1&rel=0&showinfo=0&fs=1&playsinline=1"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowfullscreen
                referrerpolicy="strict-origin-when-cross-origin">
            </iframe>
        </body>
        </html>
        """

        try:
            print(f"🌐 Embed video yükleniyor: {video_id}")
            # youtube-nocookie.com baseUrl kullanarak daha iyi uyumluluk
            self.web_view.setHtml(
                html_content, baseUrl=QUrl("https://www.youtube-nocookie.com")
            )
            self.web_view_container.setVisible(True)
            self.add_response(f"▶️ Video oynatılıyor...")

        except Exception as e:
            print(f"❌ Hata: {e}")
            self.add_response(f"❌ Video yüklenemedi: {e}")

    # --- GÜNCELLENMİŞ SLOT SONU ---

    @Slot()
    def on_turn_finished(self):
        """
        Worker'dan 'turn bitti' (hem ses hem araç çağrısı) sinyali geldiğinde tetiklenir.
        """
        print("GUI: Turn bitti sinyali alındı. Arayüz 'Hazır' durumuna getiriliyor.")
        self.update_status("Dinlemeye hazır!")

    @Slot(bool)
    def on_web_page_loaded(self, success):
        """Web sayfası yüklendiğinde çağrılır"""
        if success:
            print("✅ Web sayfası başarıyla yüklendi")
        else:
            print("❌ Web sayfası yüklenemedi")
            self.add_response("❌ Video yüklenemedi. Lütfen tekrar deneyin.")

    @Slot(str)
    def handle_error(self, error_message: str):
        self.update_status(f"Hata: {error_message}")
        timestamp = time.strftime("%H:%M:%S")
        self.response_area.append(f"[{timestamp}] ❌ HATA: {error_message}")

        self.is_listening = False
        self.mic_button.stop_listening_animation()
        self.mic_button.setText("🎤 Bas Konuş")

        if "yeniden denenecek" not in error_message:
            QMessageBox.warning(self, "Hata", error_message)

    def closeEvent(self, event):
        """Uygulama kapanırken thread'i güvenle durdurur."""
        print("Kapatma olayı tetiklendi.")
        if self.worker:
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            if not self.worker_thread.wait(3000):
                print("Thread zamanında durmadı, sonlandırılıyor.")
                self.worker_thread.terminate()
        event.accept()


def main():
    load_dotenv()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Hata: GOOGLE_API_KEY bulunamadı!")
        return

    # --- YENİ EKLENEN KISIM (CHROMIUM AYARLARI) ---
    # Bu argümanlar otomatik oynatma engelini kaldırır ve codec hatalarını azaltır
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--autoplay-policy=no-user-gesture-required "
        "--disable-web-security "
        "--allow-running-insecure-content"
    )
    app = QApplication(sys.argv)

    try:
        QWebEngineProfile.defaultProfile().setHttpCacheType(
            QWebEngineProfile.HttpCacheType.MemoryHttpCache
        )
        QWebEngineProfile.defaultProfile().setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies
        )
    except Exception as e:
        print(f"--- WEB ENGINE AYARLAMA HATASI (ÖNEMLİ OLABİLİR): {e} ---")
        sys.stdout.flush()
        # Bu hata genellikle kritik değildir, devam etmeyi deneyelim.
        pass

    window = None  # Önce None olarak tanımla
    try:
        window = EnhancedVoiceAssistantGUI()
    except Exception as e:
        # Eğer setupUI'da bir Python hatası varsa (sessiz çökme değilse)
        # burada yakalanır.
        print("\n\n---!!!! PENCERE OLUŞTURULURKEN (setupUI) HATA YAKALANDI !!!!----")
        print(f"HATA: {e}")
        import traceback

        traceback.print_exc()
        return  # Hata varsa çık

    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
