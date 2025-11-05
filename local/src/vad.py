from .utils import bytes_to_pcm16,pcm16_to_bytes
import torch
import io
import asyncio
import sounddevice as sd
import time
import numpy as np

VOICE_ENERGY_THRESHOLD = 0.015
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_TAIL_MS = 2000

class VADListener:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.samples_per_frame = 512
        self.frame_bytes = self.samples_per_frame * 2  # 16-bit mono
        self.running = False
        self.audio_q = asyncio.Queue()

        # TODO change this to Jetson Orin
        self. model , _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.model.to("cpu")

    def is_speech(self,b):
        audio = bytes_to_pcm16(b)
        rms = np.sqrt(np.mean(np.square(audio)))
        if rms < VOICE_ENERGY_THRESHOLD:
            # too quiet — likely far away or background speech
            return False
        prob = self.model(torch.from_numpy(audio), SAMPLE_RATE).item()
        #return prob > 0.5
        return prob > 0.5
    
    async def start(self):
        self.running = True
        loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status):
            try:
                mono = indata[:,0] if indata.ndim == 2 else indata
                pcm = pcm16_to_bytes(mono)
                loop.call_soon_threadsafe(self.audio_q.put_nowait, pcm)
            except Exception as e:
                print("Callback'te Hata:",e)
                pass

        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=self.sample_rate,
            dtype='float32',
            callback=callback,
            blocksize=self.samples_per_frame,
        )
        self.stream.start()

    async def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()

    async def utterances(self):
        ring = bytearray()
        speaking = False
        last_voice_time = 0.0
        utter = bytearray()
        silence_tail_s = SILENCE_TAIL_MS / 1000.0

        while self.running:
            try: # <-- Buradan başlar
                chunk = await self.audio_q.get()
                if not chunk:
                    continue
                ring.extend(chunk)

                while len(ring) >= self.frame_bytes:
                    frame = bytes(ring[:self.frame_bytes])
                    del ring[:self.frame_bytes]
                    try:
                        is_speech = self.is_speech(frame)
                    except Exception as e:
                        print("exception at is_speech:",e)
                        continue

                    now = time.time()
                    if is_speech:
                        if not speaking:
                            speaking = True
                            utter.clear()
                        # Konuşma başladığında hemen sinyal gönder: (True=Başladı, Utterance Yok)
                            yield True, None 
                    
                        utter.extend(frame)
                        last_voice_time = now

                    else:
                        if speaking and (now - last_voice_time) > silence_tail_s:
                        # Konuşma bittiğinde sinyal gönder: (False=Bitti, Utterance Var)
                            yield False, bytes(utter)
                            speaking = False
                            utter.clear()

            except Exception as e:
                # Olası bir hatayı (örneğin audio_q.get() sırasında) yakalar.
                print(f"Utterances döngüsünde kritik hata: {e}")
                # Programın çalışmasını sürdürmek için "continue" kullanabilirsiniz,
                # ancak sürekli hata veriyorsa döngüyü kırmak daha iyidir.
                break # Veya continue