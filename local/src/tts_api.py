import asyncio
import os
import io
import numpy as np
import sounddevice as sd
import soundfile as sf
from elevenlabs.client import ElevenLabs
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# API Key'i buraya girebilir veya çevre değişkenlerinden (env) alabilirsiniz.
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")


class TTSPipe:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        # ElevenLabs Client başlatma
        try:
            self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            print("[TTS] ElevenLabs Client başlatıldı.")
        except Exception as e:
            print(f"[TTS-ERROR] ElevenLabs başlatılamadı: {e}")
            self.client = None

        # Ses modelleri (Voice ID'leri ElevenLabs panelinden alabilirsiniz)
        self.voice_map = {
            "tr": "cjVigY5qzO86Huf0OWal",  # Örnek: George
            "en": "JBFqnCBsd6RMkjVDRZzb",  # Türkçe için de uygun bir model seçin
        }

    async def tts_and_play(
        self, text: str, lang: Literal["en", "tr"], timings: dict = None
    ):
        """
        Metni ElevenLabs'e gönderir, gelen sesi oynatır.
        """
        if not self.client:
            print("[TTS] Client yok, ses çalınamıyor.")
            return

        if timings and "first_tts_start" not in timings:
            timings["first_tts_start"] = asyncio.get_running_loop().time()

        try:
            voice_id = self.voice_map.get(lang, self.voice_map["en"])

            # API Çağrısı (Blocking çalışır, bu yüzden thread içinde çalıştırıyoruz)
            loop = asyncio.get_running_loop()
            audio_generator = await loop.run_in_executor(
                None,
                lambda: self.client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_turbo_v2_5",  # Düşük gecikme için Turbo model,
                    language_code=lang,
                ),
            )

            # Generator'dan tüm byte'ları topla
            audio_bytes = b"".join(audio_generator)

            if timings and "first_tts_audio_ready" not in timings:
                timings["first_tts_audio_ready"] = asyncio.get_running_loop().time()

            # Byte verisini numpy array'e çevir (SoundFile kullanarak)
            # ElevenLabs MP3 döner, soundfile bunu okuyabilir.
            with io.BytesIO(audio_bytes) as f:
                data, fs = sf.read(f, dtype="float32")

            # Ses çalma
            sd.play(data, fs, blocking=False)

            if timings and "first_audio_playback" not in timings:
                timings["first_audio_playback"] = asyncio.get_running_loop().time()

            # Çalma bitene kadar bekle
            while sd.get_stream().active:
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"[TTS-ERROR] Ses sentezleme hatası: {e}")

    def shutdown(self):
        pass
