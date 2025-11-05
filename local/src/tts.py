from transformers import VitsModel, AutoTokenizer
import kokoro
from concurrent.futures import ThreadPoolExecutor
from torchaudio.transforms import Resample
import numpy as np
import torch
import sounddevice as sd
import asyncio
import time  # <-- EKLENDİ
from typing import Literal

class TTSPipe:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.device = "cuda"
        self.tts = kokoro.KPipeline(lang_code="a", device=self.device)
        self.tr_tts_model = VitsModel.from_pretrained("mms-tts-tur", local_files_only=True).to(self.device)
        self.tr_tts_tokenizer = AutoTokenizer.from_pretrained("mms-tts-tur", local_files_only=True)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.resampler = Resample(orig_freq=16000, new_freq=24000)

    def synthesize_blocking(self, text: str, lang: Literal["tr", "en"]) -> np.ndarray:
        if lang == "en":
            result = self.tts(text, voice="af_heart")
            for data in result:
                return data.audio.cpu().numpy()
        else:
            inputs = self.tr_tts_tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.resampler(self.tr_tts_model(**inputs).waveform.cpu().squeeze(0)).numpy()
            return outputs

    # <-- METOT GÜNCELLENDİ (timings eklendi) -->
    async def tts_and_play(self, text: str, lang, timings: dict):
        loop = asyncio.get_event_loop()
        
        # 1. Sesi sentezle
        wav = await loop.run_in_executor(self.executor, self.synthesize_blocking, text, lang)
        
        # <-- EKLENDİ: Sesin hazır olma zamanı -->
        if 'first_tts_audio_ready' not in timings:
            timings['first_tts_audio_ready'] = time.monotonic()
        
        if wav is None or len(wav) == 0:
            return

        # 2. Sesi çalmaya başla
        sd.play(wav, self.sample_rate, blocking=False)
        
        # <-- EKLENDİ: Sesin çalınmaya başlama zamanı -->
        if 'first_audio_playback' not in timings:
            timings['first_audio_playback'] = time.monotonic()
        
        # 3. Sesin çalınmasını asenkron olarak bekle
        try:
            while sd.get_stream().active:
                await asyncio.sleep(0.05)
                
        except asyncio.CancelledError:
            sd.stop()
            raise 
        
    def shutdown(self):
        """Executor'ı kapatır ve bekleyen işlerin tamamlanmasını sağlar."""
        self.executor.shutdown(wait=False)