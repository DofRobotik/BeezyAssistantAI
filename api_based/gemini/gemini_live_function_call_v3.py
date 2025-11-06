import asyncio
import json
import pyaudio
from google import genai
from google.genai import types
from typing import Any, Literal, Tuple
from iot import AmrLoungeClass  # iot.py hala ışıklar için kullanılıyor
import traceback
import warnings
import requests  # YENİ: Navigasyon için eklendi
import time  # YENİ: Navigasyon payload'u için eklendi
from dotenv import load_dotenv
import os
import sys
import threading

# PTT ve çalma tarafı için durum kilidi / debounce
PTT_DEBOUNCE_MS = 200
MUTED_WHILE_RECORDING = True

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    exit()

warnings.filterwarnings("ignore")

# ------------------ CONFIG -------------------
MODEL = "gemini-live-2.5-flash-preview"
MODEL_2 = "gemini-2.5-flash-native-audio-preview-09-2025"
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# YENİ: Navigasyon Sabitleri
ROS_NAV_ENDPOINT = "http://10.10.190.14:8000/navigate"
ROBOT_ID = "amr-1"

# ---------------------------------------------

# --- IoT Cihaz Haritası (DEĞİŞİKLİK YOK) ---
iot_service_url = "10.10.10.244"
iot_port = 3001
iot = AmrLoungeClass(iot_service_url, iot_port)
light_device_map = {}
all_iot_device_codes = []
for group_index, devices in iot._AmrLoungeClass__lounge_place.items():
    for place_index, device in enumerate(devices):
        code = device["code"]
        all_iot_device_codes.append(code)
        light_device_map[code] = {"group": group_index, "index": place_index}
iot_device_prompt_list = " ,".join(all_iot_device_codes)


def execute_iot_command(target_code: str, action: str) -> Tuple[bool, str]:
    """Gerçek IoT eylemi."""
    try:
        if target_code in light_device_map:
            device_info = light_device_map[target_code]
            group = device_info["group"]
            index = device_info["index"]
            if action == "turn_on":
                iot.send_data_for_light_func(group, index, switch=True, dimming=150)
                # iot.send_data_for_light_func(dev["group"], dev["index"], True, 150)
                print(f"*** SİMÜLASYON: {target_code} AÇILDI ***")
                return True, f"{target_code} başarıyla açıldı."
            elif action == "turn_off":
                iot.send_data_for_light_func(group, index, switch=False, dimming=0)
                # iot.send_data_for_light_func(dev["group"], dev["index"], False, 0)
                print(f"*** SİMÜLASYON: {target_code} KAPATILDI ***")
                return True, f"{target_code} başarıyla kapatıldı."
            else:
                return False, f"Bilinmeyen eylem: {action}"
        return False, f"Cihaz bulunamadı: {target_code}"
    except Exception as e:
        print(f"execute_iot_command Hata: {e}")
        return False, f"Hata: {e}"


# --- YENİ: Navigasyon İstasyon Verisi ---
# (router.py'den import etmek yerine constants.py'den veriyi buraya aldık)
stations = [
    {
        "name": "station_a",
        "property": "Kitchen, a great place for drink and eat. Related to food, hunger, restaurant.",
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
station_names = [s["name"] for s in stations]
station_prompt_list = "\n".join([f"- {s['name']}: {s['property']}" for s in stations])


# --- YENİ: Navigasyon Yürütme Fonksiyonu ---
def execute_navigation_command(target_station: str) -> Tuple[bool, str]:
    """Gerçek (simüle edilmemiş) navigasyon isteğini ROS endpoint'ine gönderir."""
    if target_station not in station_names:
        print(f"*** HATA: Bilinmeyen istasyon: {target_station} ***")
        return False, f"Bilinmeyen istasyon: {target_station}"

    payload = {"station": target_station, "source": ROBOT_ID, "ts": int(time.time())}

    try:
        # 'requests' kütüphanesi senkron (blocking) olduğundan,
        # 'asyncio.to_thread' içinde çağrılacaktır.
        print(
            f"*** NAVİGASYON: {ROS_NAV_ENDPOINT} adresine {payload} gönderiliyor... ***"
        )
        response = requests.post(ROS_NAV_ENDPOINT, json=payload, timeout=5)

        response.raise_for_status()  # 2xx olmayan durumlar için hata fırlat

        print(f"*** NAVİGASYON BAŞLATILDI: {target_station} ***")
        return True, f"Navigasyon {target_station} hedefine başarıyla başlatıldı."

    except requests.exceptions.HTTPError as e:
        print(
            f"execute_navigation_command HTTP Hatası: {e.response.status_code} {e.response.text}"
        )
        return False, f"Navigasyon servisi hatası: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"execute_navigation_command Hata: {e}")
        return False, f"Navigasyon servisine bağlanılamadı: {e}"
    except Exception as e:
        print(f"execute_navigation_command Genel Hata: {e}")
        return False, f"Bilinmeyen bir hata oluştu: {e}"


# ------------------ MODEL & TOOLS -------------------

client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=GOOGLE_API_KEY)

# --- DEĞİŞİKLİK: Tool Tanımı Güncellendi (Navigasyon Eklendi) ---
tools = [
    types.Tool(
        function_declarations=[
            # 1. IoT Tool (Mevcut)
            types.FunctionDeclaration(
                name="control_iot_device",
                description="Turns on/off IoT devices such as lights. Always asks for confirmation before execution.",
                parameters={
                    "type": "object",
                    "properties": {
                        "target_device_code": {
                            "type": "string",
                            "enum": all_iot_device_codes,
                            "description": "The unique code of the device to control (e.g., 'MUTFAK_GENEL').",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["turn_on", "turn_off"],
                            "description": "The action to perform on the device.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "A brief reason why this action is being taken (e.g., 'User asked to turn on the light').",
                        },
                        "should_execute": {
                            "type": "boolean",
                            "description": "Set to 'true' ONLY if the user has explicitly confirmed the action. Otherwise, set to 'false' to ask for confirmation.",
                        },
                    },
                    "required": [
                        "target_device_code",
                        "action",
                        "reason",
                        "should_execute",
                    ],
                },
            ),
            # 2. YENİ: Navigasyon Tool'u
            types.FunctionDeclaration(
                name="navigate_to_station",
                description="Guides the robot to a specific named station (e.g., kitchen, restroom). Always asks for confirmation before execution.",
                parameters={
                    "type": "object",
                    "properties": {
                        "target_station": {
                            "type": "string",
                            "enum": station_names,
                            "description": "The unique station code to navigate to (e.g., 'station_a', 'station_b').",
                        },
                        "reason": {
                            "type": "string",
                            "description": "A brief reason why this action is being taken (e.g., 'User asked to go to the kitchen').",
                        },
                        "should_execute": {
                            "type": "boolean",
                            "description": "Set to 'true' ONLY if the user has explicitly confirmed the action. Otherwise, set to 'false' to ask for confirmation.",
                        },
                    },
                    "required": [
                        "target_station",
                        "reason",
                        "should_execute",
                    ],
                },
            ),
        ]
    )
]
turn_detection_cfg = None
try:
    # Bazı SDK sürümlerinde tip adı değişik olabilir; iki isim de deniyoruz
    LiveTurnDetection = getattr(types, "LiveTurnDetectionConfig", None) or getattr(
        types, "TurnDetectionConfig", None
    )
    if LiveTurnDetection:
        # PTT-ONLY: hiç turn detection yapma; sadece bizim PTT akışımız geçerli olsun
        turn_detection_cfg = LiveTurnDetection(type="NONE")
except Exception:
    turn_detection_cfg = None  # geriye uyumlu; alan yoksa sessizce geç

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction="You are a helpful assistant of DOF Robotics. All of your responses must be in the same language as the user's. "
    f"You can control IoT devices and navigate. "
    f"Available IoT devices: {iot_device_prompt_list}. "
    f"Available navigation stations: \n{station_prompt_list}\n"
    "## IoT Rules ## "
    "When asked to control an IoT device, you MUST first **verbally ask the user for confirmation (e.g., 'Do you confirming to turn on Kitchen Spot Lights?')** "
    "and ALSO call the 'control_iot_device' tool with 'should_execute=False'. "
    "Only after the user explicitly confirms (e.g., 'yes', 'do it'), "
    "you will call the tool again with 'should_execute=True'. "
    "## Navigation Rules ## "
    "When asked to navigate to a station, you MUST first **verbally ask for confirmation (e.g., 'Do you confirming me to guide you through Tech Shop?')** "
    "and ALSO call the 'navigate_to_station' tool with 'should_execute=False'. "
    "Only after the user explicitly confirms, "
    "you will call the tool again with 'should_execute=True'.",
    tools=tools,
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
    ),
)
# ------------------ GEMINI LIVE AGENT -------------------

pya = pyaudio.PyAudio()


class GeminiAssistant:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.is_recording = False  # PTT durumunu tutar (self.audio_stream kaldırıldı)
        self._state_lock = asyncio.Lock()
        self._last_toggle_ts = 0
        self._playback_muted = False

    # --- BU FONKSİYON SİLİNDİ ---
    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            try:
                blob = types.Blob(
                    data=msg["data"],
                    mime_type=msg.get("mime_type", "audio/pcm;rate=16000"),
                )
                await self.session.send_realtime_input(audio=blob)
            except Exception as e:
                print(f"send_realtime hatası: {e}")

    def _clear_audio_queue(self):
        if self.audio_in_queue is None:
            return
        try:
            while True:
                self.audio_in_queue.get_nowait()
                self.audio_in_queue.task_done()
        except asyncio.QueueEmpty:
            pass

    # --- GÜNCELLENDİ: control_mic (Artık buffer'ı gönderiyor) ---
    async def control_mic(self):
        """Enter ile PTT + barge-in: Enter'a bastığında modeli sustur, seni dinlesin."""
        while True:
            try:
                # 1) Konuşmaya başlamak için Enter
                await asyncio.to_thread(input, "\n🎤 Konuşmak için Enter'a basın...")

                # --- BARGE-IN NOKTASI ---
                # Model o anda konuşuyor olsa bile:
                #  - Çalan sesi sustur
                #  - Kuyrukta bekleyen tüm sesi çöpe at
                self._playback_muted = True
                self._clear_audio_queue()
                print("⏹ Model kesildi, şimdi seni dinliyorum.")

                # Eğer bir önceki kayıt doğru kapanmamışsa, güvenli şekilde kapat
                if self.is_recording:
                    print("Uyarı: Kayıt zaten aktifti. Önceki durduruluyor...")
                    self.is_recording = False
                    await asyncio.sleep(0.1)

                # Manual VAD: yeni kullanıcı aktivitesi başlıyor
                try:
                    await self.session.send_realtime_input(
                        activity_start=types.ActivityStart()
                    )
                except Exception as e:
                    print(f"activity_start gönderilemedi: {e}")

                print("🔴 Kayıt başladı... Durdurmak için Enter'a basın.")
                self.is_recording = True

                # 2) Konuşmayı bitirmek için tekrar Enter
                await asyncio.to_thread(input)

                print("⚪ Kayıt durdu. İşleniyor...")
                self.is_recording = False

                # Manual VAD: kullanıcı aktivitesi bitti
                try:
                    await self.session.send_realtime_input(
                        activity_end=types.ActivityEnd()
                    )
                except Exception as e:
                    print(f"activity_end gönderilemedi: {e}")

                # Artık model tekrar konuşabilir
                self._playback_muted = False

            except (asyncio.CancelledError, KeyboardInterrupt):
                print("Mic kontrolü iptal ediliyor.")
                self.is_recording = False
                self._playback_muted = False
                # Açık bir turn varsa kapatmayı dene (fail-safe)
                try:
                    await self.session.send_realtime_input(
                        activity_end=types.ActivityEnd()
                    )
                except Exception:
                    pass
                break
            except Exception as e:
                print(f"Mic kontrol hatası: {e}")
                self.is_recording = False
                self._playback_muted = False

    # --- GÜNCELLENDİ: listen_audio (Sadece buffer'a ekler) ---
    async def listen_audio(self):
        print("\nPTT Etkin. (Çıkış için Ctrl+C)")
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        mic_info = pya.get_default_input_device_info()

        while True:
            # PTT bekleme
            if not self.is_recording:
                await asyncio.sleep(0.01)
                continue

            stream = None
            try:
                # Stream açılışını kilit altında başlat (state konsistente kalsın)
                async with self._state_lock:
                    if not self.is_recording:
                        continue
                    stream = await asyncio.to_thread(
                        pya.open,
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=SEND_SAMPLE_RATE,
                        input=True,
                        input_device_index=mic_info["index"],
                        frames_per_buffer=CHUNK_SIZE,
                    )
                    print("Stream açıldı, dinleniyor...")

                # Okuma döngüsü
                while self.is_recording:
                    try:
                        data = await asyncio.to_thread(
                            stream.read, CHUNK_SIZE, **kwargs
                        )
                        await self.out_queue.put(
                            {"data": data, "mime_type": "audio/pcm"}
                        )
                    except IOError as e:
                        if getattr(e, "errno", None) == pyaudio.paInputOverflowed:
                            print("Uyarı: Input Overflowed. Chunk atlanıyor.")
                            continue
                        print(f"Mic okuma hatası (IOError): {e}")
                        break

            except Exception as e:
                print(f"Bilinmeyen listen_audio hatası: {e}")
                traceback.print_exc()
            finally:
                # Stream'i kesin ve güvenli kapat
                if stream:
                    try:
                        await asyncio.to_thread(stream.stop_stream)
                    except Exception:
                        pass
                    try:
                        await asyncio.to_thread(stream.close)
                    except Exception:
                        pass
                    print("Stream kapatıldı.")

    # --- receive_audio (DEĞİŞİKLİK YOK) ---
    # Bu fonksiyon AI'dan gelen sesi (output) yönetir,
    # bizim PTT (input) değişikliğimizden etkilenmez.

    async def receive_audio(self):
        """
        Gelen yanıtları dinler, sesi oynatır ve
        hem IoT hem de Navigasyon araç çağrılarını (function call) işler.
        """

        while True:
            try:
                turn = self.session.receive()
                async for chunk in turn:
                    # 1. Sunucu İçeriği (Ses veya Metin)
                    if chunk.server_content:
                        if data := chunk.data:
                            self.audio_in_queue.put_nowait(data)
                        if text := chunk.text:
                            print(text, end="")

                    # 2. Araç Çağrısı (Function Call)
                    elif chunk.tool_call:
                        print(f"\n[🔄 Araç Çağrısı Algılandı]")
                        function_responses_to_send = []
                        for fc in chunk.tool_call.function_calls:
                            print(
                                f"[İşleniyor: {fc.name}, ID: {fc.id}, Args: {fc.args}]"
                            )
                            try:
                                args = fc.args
                                should_execute = args.get("should_execute", False)

                                if not should_execute:
                                    print(
                                        f"❓ Model '{fc.name}' için onay istiyor. Kullanıcı yanıtı bekleniyor."
                                    )
                                    continue  # Onaylanmamışsa bir sonraki fonksiyona geç

                                # --- Yürütme Mantığı ---

                                response_data = {
                                    "success": False,
                                    "message": "Bilinmeyen fonksiyon",
                                }

                                # Durum 1: IoT Cihaz Kontrolü
                                if fc.name == "control_iot_device":
                                    target = args.get("target_device_code")
                                    action = args.get("action")
                                    print(
                                        f"✅ Onay alındı. IoT Cihazı: {target} için '{action}' eylemi yürütülüyor..."
                                    )
                                    # IoT komutu senkron, ama hızlı çalışıyor (simülasyon)
                                    # Gerçek dünyada 'asyncio.to_thread' gerekebilir
                                    success, message = execute_iot_command(
                                        target, action
                                    )
                                    response_data = {
                                        "success": success,
                                        "message": message,
                                    }

                                # Durum 2: Navigasyon
                                elif fc.name == "navigate_to_station":
                                    target = args.get("target_station")
                                    print(
                                        f"✅ Onay alındı. Navigasyon: {target} hedefine yönlendiriliyor..."
                                    )
                                    # HTTP isteği (requests) blocking'dir, bu yüzden 'to_thread' kullanıyoruz
                                    success, message = await asyncio.to_thread(
                                        execute_navigation_command, target
                                    )
                                    response_data = {
                                        "success": success,
                                        "message": message,
                                    }

                                else:
                                    print(f"❌ Bilinmeyen fonksiyon adı: {fc.name}")

                                # --- Yürütme Bitti ---

                                if response_data["success"]:
                                    print(f"İşlem başarılı: {response_data['message']}")
                                else:
                                    print(
                                        f"İşlem başarısız: {response_data['message']}"
                                    )

                                function_responses_to_send.append(
                                    types.FunctionResponse(
                                        id=fc.id, name=fc.name, response=response_data
                                    )
                                )
                            except Exception as e:
                                print(f"❌ Fonksiyon işleme hatası: {e}")
                                traceback.print_exc()
                                function_responses_to_send.append(
                                    types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"success": False, "message": str(e)},
                                    )
                                )

                        if function_responses_to_send:
                            print(
                                f"[📬 {len(function_responses_to_send)} adet fonksiyon yanıtı gönderiliyor...]"
                            )
                            await self.session.send_tool_response(
                                function_responses=function_responses_to_send
                            )

                    # 3. Diğer (Kesinti vb. - Bu kodda işlenmiyor)

            except asyncio.CancelledError:
                print("receive_audio (while True) iptal edildi.")
                break
            except Exception as e:
                print(f"Hata: 'receive_audio' ana akışında sorun oluştu: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
                continue

    # --- play_audio (DEĞİŞİKLİK YOK) ---
    # Bu da AI output'u ile ilgili, değişikliğe gerek yok.
    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()

            # Barge-in veya kayıt sırasında model sesini çalma
            if self._playback_muted or self.is_recording:
                continue

            await asyncio.to_thread(stream.write, bytestream)

    # --- GÜNCELLENDİ: run (out_queue ve send_realtime kaldırıldı) ---
    async def run(self):
        tasks = set()  # Görevleri takip etmek için bir set
        try:
            async with client.aio.live.connect(model=MODEL_2, config=CONFIG) as session:
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=100)

                # Görevleri oluştur ve sete ekle
                tasks.add(asyncio.create_task(self.send_realtime()))
                tasks.add(asyncio.create_task(self.listen_audio()))
                tasks.add(asyncio.create_task(self.control_mic()))
                tasks.add(asyncio.create_task(self.receive_audio()))
                tasks.add(asyncio.create_task(self.play_audio()))

                # Tüm görevlerin tamamlanmasını bekle (veya birinin hata vermesini)
                await asyncio.gather(*tasks)

        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\nProgram sonlandırılıyor...")
            # Hata veya kesinti durumunda 'finally' bloğu çalışacak

        except Exception as e:
            # Görevlerden herhangi birinde oluşan beklenmedik hataları yakala
            print(f"Ana 'run' döngüsünde beklenmedik bir hata oluştu: {e}")
            traceback.print_exc()
            # Hata durumunda 'finally' bloğu çalışacak

        finally:
            # Program sonlanırken (hata, kesinti veya normal çıkış)
            # tüm görevlerin düzgünce iptal edildiğinden emin ol.
            print("Tüm görevler iptal ediliyor...")
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Görevlerin iptal işlemini tamamlaması için bekle
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except asyncio.CancelledError:
                    pass  # Kapatma sırasında bu beklenir

            # Kaynakları temizle
            # 'self.audio_stream' kontrolü kaldırıldı, çünkü artık 'listen_audio'
            # kendi stream'ini lokal olarak yönetiyor ve kapatıyor.
            pya.terminate()
            print("Kaynaklar temizlendi. Çıkıldı.")


# ------------------ MAIN (DEĞİŞİKLİK YOK) -------------------

if __name__ == "__main__":
    try:
        asyncio.run(GeminiAssistant().run())
    except KeyboardInterrupt:
        print("\nÇıkış yapıldı.")
