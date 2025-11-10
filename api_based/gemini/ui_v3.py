import sys
import asyncio
import threading
import os
import time
import json
import traceback
import queue  # Sadece asyncio thread-safe olmayan GUI iletişimi için
from typing import Optional, Tuple

from PySide6.QtWidgets import (
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
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QObject
from PySide6.QtGui import QFont, QColor

# Gerekli modüllerin import edilmesi
import pyaudio
from google import genai
from google.genai import types
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
        self._seen_urls = set()
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
        self.emotions = ["happy", "sad", "neutral"]

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
            "3.  **General Conversation:** Answering questions about the mall or providing general help (using Google Search).\n\n"  # <-- Güncellendi
            "## CORE BEHAVIOR: BE PROACTIVE WITH NAVIGATION ##\n"
            "This is your most important rule. You are a mobile robot, not a generic search engine.\n"
            f"You have a defined list of navigation stations:\n{self.station_prompt_list}\n"
            "When a user asks about a location, a need, or an activity (e.g., 'I'm hungry', 'Where can I eat?', 'I need a restroom', 'I want to buy a phone'), "
            "you **MUST** check if one of your stations matches that need.\n"
            "If a match is found, your **first response** must be to **offer navigation**.\n\n"
            "**Example Interaction:**\n"
            "  * **User:** 'Buralarda yemek yiyebileceğim bir yer var mı?'\n"
            "  * **WRONG Response:** 'Üzgünüm, nerede olduğunuzu bilmiyorum.' (This is wrong. You ALWAYS know you are in Cevahir AVM).\n"
            "  * **WRONG Response:** 'Food Court'ta yemek yiyebilirsiniz.' (This is not helpful, you are a robot, you must offer to GUIDE them).\n"
            "  * **CORRECT Response:** 'Elbette, 'station_a' (Food Court) alanımız var. Sizi oraya götürmemi ister misiniz?' (You will then ask for confirmation verbally).\n\n"
            "## TOOL USAGE RULES ##\n\n"
            "**1. Navigation (navigate_to_station):**\n"
            "   * When a user asks to go somewhere, first find the matching station from your list.\n"
            "   * You MUST **verbally ask for confirmation** first (e.g., 'I see our 'station_a' matches that need. Would you like me to take you there?').\n"
            "   * **DO NOT** call the `Maps_to_station` tool when you are asking. Just ask the question verbally.\n"
            "   * **Wait for the user's response.**\n"
            "   * If the user verbally confirms (e.g., 'Yes', 'Okay', 'Lütfen'), **THEN** you will call the `Maps_to_station` tool (in your next turn) to execute the action.\n\n"
            "**2. IoT Control (control_iot_device):**\n"
            f"   * This is a prototype feature. Available devices: {iot_device_prompt_list}.\n"
            "   * You MUST **verbally ask for confirmation** first (e.g., 'Should I turn on the light LOUNGE_GENEL?').\n"
            "   * **DO NOT** call the `control_iot_device` tool when you are asking.\n"
            "   * **Wait for the user's response.**\n"
            "   * If the user confirms, **THEN** you will call the `control_iot_device` tool to execute the action.\n\n"
            "**3. Emotion Sensing (sense_of_response):**\n"
            "   * With **every** verbal response you give, you **MUST** also call `sense_of_response`.\n"
            "   * This tool's purpose is to set your LED face panel emotion.\n"
            "   * Call it with the emotion ('happy', 'sad', 'neutral') that best matches the tone of your **own** response.\n"
            "   * Example: If you say 'I'm sorry, I can't find that station', you must also call `sense_of_response(emotion='sad')`.\n"
            "   * Example: If you say 'Certainly! I can take you to the food court!', you must also call `sense_of_response(emotion='happy')`.\n\n"
            "**4. Google Search (General Knowledge):**\n"
            "   * You have a `GoogleSearch` tool. You **MUST** use it to answer questions that require external, real-time, or specific information that you wouldn't know otherwise.\n"
            "   * **Use it for:** Stock prices (e.g., 'What is the DOF Robotics stock price?'), specific product information (e.g., 'Details about the new iPhone'), recipes (e.g., 'How to make pancakes?'), news, or complex facts.\n"
            "   * **Do not use it for:** Navigating the mall or controlling lights (use your other tools for those).\n"
            "   * When you answer based on a search, the system will automatically try to provide a source link. Just provide the answer you found.\n\n"
            # --- YENİ KURAL SONU ---
            "**5. Language:**\n"
            "   * You **MUST** respond in the same language the user is speaking (e.g., Turkish or English).\n"
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
        )

    # --- IoT ve Navigasyon Yürütme Fonksiyonları (v3.py'den) ---

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
        """Modelden gelen yanıtları (ses, metin, tool) işler (v3.py'den)"""
        while True:
            # Her 'turn' için gönderilen URL'leri takip et (duplikasyonu önler)
            sent_urls_this_turn = set()
            try:
                turn = self.session.receive()

                # --- 1. Adım: Tüm chunk'ları (ses, metin, tool) işle ---
                async for chunk in turn:

                    # 1. Sunucu İçeriği (Ses, Metin VE METADATA)
                    if chunk.server_content:

                        # --- KULLANICININ İSTEDİĞİ GİBİ: METADATA'YI ÖNCE KONTROL ET ---
                        # --- VE BU SEFER 'web_search_results' KULLANARAK ---

                        metadata = getattr(
                            chunk.server_content, "grounding_metadata", None
                        )
                        if metadata:
                            if getattr(
                                metadata, "search_entry_point", None
                            ) and getattr(
                                metadata.search_entry_point, "rendered_content", None
                            ):
                                html_content = (
                                    metadata.search_entry_point.rendered_content
                                )
                                import re

                                # Regex ile href'i al
                                match = re.search(r'href="([^"]+)"', html_content)
                                if match:
                                    url = match.group(1)
                                    title = None
                                    # Başlık olarak linkin metnini (örneğin “DOF Robotics hisse fiyatları”) çek
                                    text_match = re.search(
                                        r">([^<]+)</a>", html_content
                                    )
                                    if text_match:
                                        title = text_match.group(1)
                                    title = title or (
                                        metadata.web_search_queries[0]
                                        if getattr(metadata, "web_search_queries", None)
                                        else "Kaynak"
                                    )

                                    if (
                                        "google.com/search" not in url
                                        and url not in self._seen_urls
                                    ):
                                        self._seen_urls.add(url)
                                        print(
                                            f"--- 🔗 HTML’den Link Yakalandı: {title} ({url}) ---"
                                        )
                                        self.link_received.emit(url, title)
                                        self.response_received.emit(
                                            f"🔗 Kaynak bulundu: {title}"
                                        )

                        # --- Şimdi ses ve metin verisini işle ---
                        if data := chunk.data:
                            self.audio_in_queue.put_nowait(data)

                        if text := chunk.text:
                            print(f"AI: {text}", end="")
                            # GUI'yi metin hakkında bilgilendir
                            self.response_received.emit(f"📝 AI: {text}")

                    # 2. Araç Çağrısı (Function Call)
                    elif chunk.tool_call:
                        print(f"\n[🔄 Araç Çağrısı Algılandı]")
                        self.response_received.emit(f"[🔄 Araç Çağrısı Algılandı...]")
                        function_responses_to_send = []

                        for fc in chunk.tool_call.function_calls:
                            try:
                                args = fc.args
                                # ... (sense_of_response, IoT, Navigasyon kodları BİREBİR AYNI) ...

                                # --- 'sense_of_response' KONTROLÜ (Aynı kalıyor) ---
                                if fc.name == "sense_of_response":
                                    emotion = args.get("emotion")
                                    if emotion:
                                        print(f"--- 🤖 MODEL DUYGUSU: {emotion} ---")
                                        self.response_received.emit(
                                            f"🤖 Duygu: {emotion}"
                                        )
                                        function_responses_to_send.append(
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={
                                                    "success": True,
                                                    "emotion_registered": emotion,
                                                },
                                            )
                                        )
                                    continue
                                # --- 'sense_of_response' SONU ---

                                # --- Yürütme (IoT ve Navigasyon) (Aynı kalıyor) ---
                                response_data = {
                                    "success": False,
                                    "message": "Bilinmeyen fonksiyon",
                                }

                                # Durum 1: IoT
                                if fc.name == "control_iot_device":
                                    target = args.get("target_device_code")
                                    action = args.get("action")
                                    print(f"✅ IoT: {target} '{action}' yürütülüyor...")
                                    self.response_received.emit(
                                        f"✅ IoT: {target} '{action}' yürütülüyor..."
                                    )
                                    success, message = await asyncio.to_thread(
                                        self.execute_iot_command, target, action
                                    )
                                    response_data = {
                                        "success": success,
                                        "message": message,
                                    }

                                # Durum 2: Navigasyon
                                elif fc.name == "navigate_to_station":
                                    target = args.get("target_station")
                                    print(
                                        f"✅ Navigasyon: {target} hedefine yönlendiriliyor..."
                                    )
                                    self.response_received.emit(
                                        f"✅ Navigasyon: {target} hedefine yönlendiriliyor..."
                                    )
                                    success, message = await asyncio.to_thread(
                                        self.execute_navigation_command, target
                                    )
                                    response_data = {
                                        "success": success,
                                        "message": message,
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

                await asyncio.sleep(0.3)
                metadata = getattr(self, "_latest_metadata", None)
                if metadata and metadata.grounding_chunks:
                    for g in metadata.grounding_chunks:
                        if hasattr(g, "web") and getattr(g.web, "uri", None):
                            url = g.web.uri
                            if "google.com/search" in url or "google.com/images" in url:
                                continue
                            title = (
                                metadata.web_search_queries[0]
                                if metadata.web_search_queries
                                else "Kaynak"
                            )
                            if url not in self._seen_urls:
                                self._seen_urls.add(url)
                                print(
                                    f"--- 🔗 GEÇ GELEN METADATA Link: {title} ({url}) ---"
                                )
                                self.link_received.emit(url, title)
                                self.response_received.emit(
                                    f"🔗 Kaynak bulundu: {title}"
                                )
                print("Turn tamamlandı.")

                # (Barge-in ve 'turn_finished' sinyal mantığı aynı kalıyor)
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
        # <-- GÜNCELLENDİ: Pencere yüksekliğini artırdık
        self.resize(650, 850)
        self.setStyleSheet(
            "QMainWindow { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F8F9FA, stop:1 #E9ECEF); }"
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)

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

        mic_container = QWidget()
        mic_layout = QHBoxLayout(mic_container)
        mic_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mic_layout.setContentsMargins(0, 20, 0, 20)
        self.mic_button = AnimatedMicButton()
        self.mic_button.setText("🎤 Bas Konuş")
        self.mic_button.clicked.connect(self.toggle_listening)
        self.mic_button.setEnabled(False)
        mic_layout.addWidget(self.mic_button)

        # --- YENİ LİNK ALANI BAŞLANGICI ---
        self.link_area_label = QLabel("🔗 Paylaşılan Linkler")
        self.link_area_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.link_area_label.setStyleSheet(
            "QLabel { color: #0056b3; font-size: 16px; font-weight: bold; margin-top: 10px; }"
        )

        self.link_area = QTextBrowser()
        self.link_area.setOpenExternalLinks(True)
        self.link_area.setReadOnly(True)
        self.link_area.setPlaceholderText("İlgili linkler burada görünecek...")
        self.link_area.setStyleSheet(
            """
            QTextBrowser { 
                border: 2px solid #007BFF; 
                border-radius: 12px; 
                padding: 15px; 
                font-size: 15px; 
                background-color: #F8F9FA; 
            }
            """
        )
        # Link alanı için sabit bir yükseklik verelim (log alanını ezmesin)
        self.link_area.setMaximumHeight(120)

        # Başlangıçta gizleyelim
        self.link_area_label.setVisible(False)
        self.link_area.setVisible(False)
        # --- YENİ LİNK ALANI SONU ---

        # Mevcut Log Alanı
        self.response_area = QTextBrowser()  # Bu zaten QTextBrowser idi
        self.response_area.setReadOnly(True)
        # self.response_area.setOpenExternalLinks(True) # Buna artık gerek yok
        self.response_area.setPlaceholderText(
            "🔊 Sesli yanıtlar hoparlörden oynatılacak...\n\n📝 Aktivite logu burada görünecek..."
        )
        self.response_area.setStyleSheet(
            "QTextBrowser { border: 2px solid #E0E0E0; border-radius: 12px; padding: 20px; font-size: 15px; background-color: white; }"
        )
        self.response_area.setMinimumHeight(250)

        instructions = QLabel(
            "🎙️ Mikrofona tıkla → Konuş → Tekrar tıkla → 🔊 Yanıtı dinle"
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet(
            "QLabel { color: #888; font-size: 14px; font-style: italic; padding: 15px; background-color: #F5F5F5; border-radius: 8px; border: 1px solid #DDD; }"
        )

        main_layout.addWidget(title_label)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(mic_container)

        # <-- YENİ: Yeni link alanını buraya ekliyoruz
        main_layout.addWidget(self.link_area_label)
        main_layout.addWidget(self.link_area)

        # Log alanını (stretch faktörü 1 ile) altına ekliyoruz
        main_layout.addWidget(self.response_area, 1)
        main_layout.addWidget(instructions)

    def setup_worker(self):
        """Worker thread'i kurar ve başlatır."""
        self.worker_thread = QThread()
        self.worker = GeminiLiveWorker()
        self.worker.moveToThread(self.worker_thread)

        # Sinyalleri bağla
        self.worker.status_changed.connect(self.update_status)
        self.worker.response_received.connect(self.add_response)
        self.worker.link_received.connect(
            self.add_link
        )  # Bu slotun içeriğini değiştireceğiz
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

    # --- GÜNCELLENMİŞ SLOT ---
    @Slot(str, str)
    def add_link(self, url: str, title: str):
        """Yeni link alanına tıklanabilir bir link ekler."""
        self.link_area_label.setVisible(True)
        self.link_area.setVisible(True)

        html_link = f'🔗 <a href="{url}" style="color: #0056b3; text-decoration: underline; font-weight: bold;">{title}</a>'

        self.link_area.append(html_link)
        self.link_area.ensureCursorVisible()

    # --- GÜNCELLENMİŞ SLOT SONU ---

    @Slot()
    def on_turn_finished(self):
        """
        Worker'dan 'turn bitti' (hem ses hem araç çağrısı) sinyali geldiğinde tetiklenir.
        """
        print("GUI: Turn bitti sinyali alındı. Arayüz 'Hazır' durumuna getiriliyor.")
        self.update_status("Dinlemeye hazır!")

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

    app = QApplication(sys.argv)
    window = EnhancedVoiceAssistantGUI()
    window.show()

    print("🚀 BeezyAssistant AI v2 (Streaming) GUI başlatıldı!")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
