import asyncio
import json
import pyaudio
from google import genai
from google.genai import types
from typing import Any, Literal, Tuple
from iot import AmrLoungeClass
import traceback
import warnings

warnings.filterwarnings("ignore")


# ------------------ CONFIG -------------------
MODEL = "gemini-live-2.5-flash-preview"  # veya en güncel preview modeli
MODEL_2 = "gemini-2.5-flash-native-audio-preview-09-2025"
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
# ---------------------------------------------

# --- DEĞİŞİKLİK 1: Sistem Talimatı Güncellendi ---

# IoT cihaz haritası çıkarımı
iot = AmrLoungeClass()
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
            dev = light_device_map[target_code]
            if action == "turn_on":
                # iot.send_data_for_light_func(dev["group"], dev["index"], True, 150)
                print(f"*** SİMÜLASYON: {target_code} AÇILDI ***")
                return True, f"{target_code} başarıyla açıldı."
            elif action == "turn_off":
                # iot.send_data_for_light_func(dev["group"], dev["index"], False, 0)
                print(f"*** SİMÜLASYON: {target_code} KAPATILDI ***")
                return True, f"{target_code} başarıyla kapatıldı."
            else:
                return False, f"Bilinmeyen eylem: {action}"
        return False, f"Cihaz bulunamadı: {target_code}"
    except Exception as e:
        print(f"execute_iot_command Hata: {e}")
        return False, f"Hata: {e}"


# ------------------ MODEL & TOOLS -------------------

client = genai.Client(http_options={"api_version": "v1beta"}, api_key=GOOGLE_API_KEY)

# --- DEĞİŞİKLİK 2: Tool Tanımı Güncellendi ---
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="control_iot_device",
                description="Turns on/off IoT devices such as lights. Always asks for confirmation before execution.",
                parameters={
                    "type": "object",
                    "properties": {
                        "target_device_code": {
                            "type": "string",
                            "enum": all_iot_device_codes,
                            "description": "The unique code of the device to control (e.g., 'MUTFAK_GENEL', 'AMFI_SPOT').",
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
                    # 'should_execute' artık zorunlu bir alan
                    "required": [
                        "target_device_code",
                        "action",
                        "reason",
                        "should_execute",
                    ],
                },
            )
        ]
    )
]
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction="You are a helpful assistant of DOF Robotics. All of your responses must be in the same language as the user's. "
    "When asked to control an IoT device, you MUST first **verbally ask the user for confirmation (e.g., 'Mutfak ışığını açmamı onaylıyor musun?')** "
    "and ALSO call the 'control_iot_device' tool with 'should_execute=False'. "
    "Only after the user explicitly confirms (e.g., 'yes', 'do it'), "
    "you will call the tool again with 'should_execute=True'.",
    tools=tools,
)

# ------------------ GEMINI LIVE AGENT -------------------

pya = pyaudio.PyAudio()


class GeminiAssistant:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None  # audio_stream'i burada tanımla

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}

        print("\n🎤 Dinliyorum... (Çıkış için Ctrl+C)")
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    # --- DEĞİŞİKLİK 3: 'receive_audio' Metodu Tamamen Değiştirildi ---
    async def receive_audio(self):
        """
        'while True' döngüsü ve doğru 'chunk' isimleri
        (interruption_metadata, turn_complete_metadata) ile düzeltilmiş versiyon.
        """

        while True:  # Her 'turn' için yeniden dinlemeyi başlatan ana döngü
            try:
                # 'session.receive()' her 'turn' için yeni bir dinleyici başlatır.
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
                                        "❓ Model onay istiyor. Kullanıcı yanıtı bekleniyor."
                                    )
                                    continue

                                target = args.get("target_device_code")
                                action = args.get("action")
                                print(
                                    f"✅ Onay alındı. {target} için '{action}' eylemi yürütülüyor..."
                                )

                                success, message = execute_iot_command(target, action)
                                response_data = {"success": success, "message": message}
                                if success:
                                    print("Operation done successfully.")
                                else:
                                    print("Unsuccessfull.")

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

                    # --- DÜZELTME BURADA ---
                    # 3. Kesinti (Kullanıcı araya girdi)

            except asyncio.CancelledError:
                print("receive_audio (while True) iptal edildi.")
                break
            except Exception as e:
                print(f"Hata: 'receive_audio' ana akışında sorun oluştu: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
                continue

    # --- DEĞİŞİKLİK 3 BİTTİ ---

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
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(
                    model=MODEL_2, config=CONFIG
                ) as session,  # 'tools' buraya eklendi
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=100)

                send_text_task = tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("Kullanıcı çıkış yaptı")

        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\nProgram sonlandırılıyor...")
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            # Kaynakları temizle
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            pya.terminate()
            print("Kaynaklar temizlendi. Çıkıldı.")


# ------------------ MAIN -------------------

if __name__ == "__main__":
    asyncio.run(GeminiAssistant().run())
