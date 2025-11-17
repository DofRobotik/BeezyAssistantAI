import asyncio
import json
import re
import time
import httpx  # ROS isteği için gerekli (pip install httpx)
from concurrent.futures import ThreadPoolExecutor

# Ollama Async Client
from ollama import AsyncClient

# Local modules
from stt import STTPipe
from tts_api import TTSPipe
from search_tool import search_web
from tools import get_tools_schema
from prompts import get_system_prompt
from constants import stations

try:
    from iot import AmrLoungeClass
except ImportError:
    print("IoT module not found, running in mock mode.")
    AmrLoungeClass = None


class AssistantApp:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks if callbacks else {}

        # 1. Model Config
        self.model_name = "qwen2.5:14b"
        self.client = AsyncClient()

        # 2. Navigasyon Ayarları (ROS)
        self.ros_nav_endpoint = "http://10.10.190.14:8000/navigate"
        self.robot_id = "amr-1"

        # 3. Bileşenler
        self.stt = STTPipe("faster-whisper-large-v3")
        self.tts = TTSPipe()

        # 4. IoT Kurulumu
        self.iot_controller = None
        self.device_codes = []
        self.device_map = {}

        if AmrLoungeClass:
            try:
                self.iot_controller = AmrLoungeClass("10.10.10.244", 3001)
                self.device_map = self.iot_controller.get_all_devices()
                self.device_codes = list(self.device_map.keys())
            except Exception as e:
                print(f"IoT Connection Failed: {e}")

        # 5. Prompt ve Araçlar
        self.station_str = "\n".join(
            [f"- {s['name']}: {s['property']}" for s in stations]
        )
        self.device_str = ", ".join(self.device_codes)

        self.system_prompt = get_system_prompt(self.station_str, self.device_str)

        self.tools = get_tools_schema([s["name"] for s in stations], self.device_codes)

        # 6. Geçmiş ve State
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        self.stt_executor = ThreadPoolExecutor(max_workers=1)
        self.is_thinking = False

    def _trigger(self, event, *args):
        if event in self.callbacks:
            self.callbacks[event](*args)

    # --- TOOL FONKSİYONLARI ---

    async def _tool_search(self, query):
        self._trigger("on_response_chunk", f"\n🔎 Searching: {query}...\n")
        return await asyncio.to_thread(search_web, query, 4, "tr")

    async def _tool_navigate(self, station_name):
        """Gerçek ROS Köprüsüne İstek Atar"""
        print(f"🚀 NAVIGATING TO: {station_name} via {self.ros_nav_endpoint}")

        payload = {
            "station": station_name,
            "source": self.robot_id,
            "ts": int(time.time()),
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(self.ros_nav_endpoint, json=payload)

            if 200 <= resp.status_code < 300:
                return f"Navigation successfully started to {station_name}. (Status: {resp.status_code})"
            else:
                return f"Navigation Request Failed. ROS Status: {resp.status_code}, Error: {resp.text}"

        except Exception as e:
            print(f"[ROS ERROR] {e}")
            return f"Connection error to Robot Navigation System: {e}"

    def _tool_iot(self, device_code, action):
        if not self.iot_controller or device_code not in self.device_map:
            return "Error: Device not found or IoT system offline."

        dev = self.device_map[device_code]
        switch = True if action == "turn_on" else False

        if dev["type"] == "light":
            # Blocking olabilir ama çok hızlı olduğu için sorun olmayabilir.
            # Yine de garanti olsun diye thread'e alınabilir, ama şimdilik böyle kalsın.
            self.iot_controller.send_data_for_light_func(
                dev["group"], dev["index"], switch
            )
            return f"Device {device_code} turned {action}."

        return "Device type not fully supported yet."

    # --- ANA MANTIK ---

    async def _handle_conversation(self, user_text: str):
        self.chat_history.append({"role": "user", "content": user_text})
        print(f"[LLM] User: {user_text}")

        self._trigger("on_response_start")

        # 1. Tool Kararı
        response = await self.client.chat(
            model=self.model_name,
            messages=self.chat_history,
            tools=self.tools,
        )

        message = response["message"]
        tool_calls = message.get("tool_calls")

        if tool_calls:
            self.chat_history.append(message)

            for tool in tool_calls:
                fn_name = tool["function"]["name"]
                args = tool["function"]["arguments"]
                print(f"[TOOL] Calling {fn_name} with {args}")

                result_content = ""

                if fn_name == "search_web":
                    result_content = await self._tool_search(args.get("query"))
                elif fn_name == "navigate_to_station":
                    # ARTIK ASYNC ÇAĞRILIYOR (ROS İsteği burada)
                    result_content = await self._tool_navigate(args.get("station_name"))
                elif fn_name == "control_iot_device":
                    result_content = self._tool_iot(
                        args.get("device_code"), args.get("action")
                    )

                self.chat_history.append(
                    {
                        "role": "tool",
                        "content": str(result_content),
                    }
                )

            # 2. Final Cevap
            print("[LLM] Generating final response after tool execution...")
            async for chunk in await self.client.chat(
                model=self.model_name, messages=self.chat_history, stream=True
            ):
                await self._process_stream_chunk(chunk)

        else:
            # Tool Yoksa
            content = message["content"]
            self.chat_history.append({"role": "assistant", "content": content})

            clean_text = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()

            if clean_text:
                self._trigger("on_response_chunk", clean_text)
                await self.tts.tts_and_play(clean_text, "tr")

        self._trigger("on_response_end")

    async def _process_stream_chunk(self, chunk):
        content = chunk["message"]["content"]

        # <think> Filtreleme
        if "<think>" in content:
            self.is_thinking = True
            content = content.replace("<think>", "")

        if "</think>" in content:
            self.is_thinking = False
            content = content.replace("</think>", "")
            parts = content.split("</think>")
            if len(parts) > 1:
                content = parts[1]

        if self.is_thinking or not content:
            return

        self._trigger("on_response_chunk", content)
        await self.tts_buffer_logic(content)

    _tts_buffer = ""

    async def tts_buffer_logic(self, text_chunk):
        self._tts_buffer += text_chunk
        if any(p in text_chunk for p in [".", "?", "!", "\n"]):
            to_speak = self._tts_buffer.strip()
            if to_speak:
                await self.tts.tts_and_play(to_speak, "tr")
            self._tts_buffer = ""

    # --- AUDIO HANDLERS ---
    def start_listening(self):
        self.is_recording = True
        self.audio_buffer = []
        self._trigger("on_listening_started")
        import sounddevice as sd

        def cb(indata, f, t, s):
            if self.is_recording:
                self.audio_buffer.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="int16", callback=cb
        )
        self.stream.start()

    async def stop_listening(self):
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        import numpy as np

        if not self.audio_buffer:
            self._trigger("on_ready")
            return

        audio = np.concatenate(self.audio_buffer, axis=0).tobytes()
        self._trigger("on_processing_started", "Transcribing...")

        text, lang = await asyncio.get_event_loop().run_in_executor(
            self.stt_executor, self.stt.stt, audio
        )

        if text and len(text) > 1:
            self._trigger("on_transcription_done", text)
            await self._handle_conversation(text)
        else:
            self._trigger("on_error", "Ses anlaşılamadı.")

        self._trigger("on_ready")

    async def run(self):
        self._trigger("on_ready")
        while True:
            await asyncio.sleep(1)
