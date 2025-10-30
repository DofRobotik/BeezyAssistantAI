import sys
import asyncio
import re
import time
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor

from .llm import LLM
from .vad import VADListener
from .stt import STTPipe
from .tts import TTSPipe
from .router import Agent
from .constants import stations
try:
    from .iot import AmrLoungeClass
except ImportError:
    print("Warning: .iot import failed, trying direct import 'iot'")
    from iot import AmrLoungeClass


class AssistantApp:
    def __init__(self):
        self.vad = VADListener()
        
        # --- STT MODEL TAVSİYESİ ---
        # 7 saniyelik gecikmeyi çözmek için 'large-v3' yerine 'medium' veya 'base' kullanın.
        # self.stt = STTPipe("faster-whisper-large-v3-turbo-ct2") # ÇOK YAVAŞ
        self.stt = STTPipe("faster-whisper-large-v3") # TAVSİYE EDİLEN
        # self.stt = STTPipe("faster-whisper-base-ct2") # ÇOK HIZLI (doğruluk düşebilir)
        # --- TAVSİYE BİTTİ ---
        
        self.llm = LLM(model="qwen3:8b")
        self.tts = TTSPipe()

        try:
            iot_service_url = "10.10.10.244" # (Lütfen doğru IP'yi buraya girin)
            iot_port = 3001
            self.iot = AmrLoungeClass(iot_service_url, iot_port)
            print(f"[ASSISTANT] IoT Servisi başlatıldı. URL: {iot_service_url}:{iot_port}")
        except Exception as e:
            print(f"[ASSISTANT-ERROR] IoT Servisi başlatılamadı: {e}")
            self.iot = None

        self.agent = Agent(
            llm=self.llm, 
            stations=stations, # constants.py'den gelen liste
            iot_controller=self.iot, 
            robot_id="amr-1"
        )

        # confirmation state
        self.awaiting_confirmation: bool = False
        self.pending_station: Optional[str] = None

        # Executor'lar
        self.stt_executor = ThreadPoolExecutor(max_workers=1)
        self.llm_executor = ThreadPoolExecutor(max_workers=1)
        self.iot_executor = ThreadPoolExecutor(max_workers=1)

        # BARGE-IN YÖNETİMİ
        self.tts_task: Optional[asyncio.Task] = None
        self.current_response_task: Optional[asyncio.Task] = None

    async def say(self, text: str, lang, timings: dict):
        """Asistanın konuşmasını başlatır ve görevi tts_task'ta saklar."""
        
        # Sadece ilk çağrıda 'TTS Sentez Başlangıcı' zaman damgasını al
        if 'first_tts_start' not in timings:
            timings['first_tts_start'] = time.monotonic()
            
        self.tts_task = asyncio.create_task(self.tts.tts_and_play(text, lang, timings))
        try:
            await self.tts_task
        except asyncio.CancelledError:
            print("[ASSISTANT] Konuşma iptal edildi (Barge-in).")
            raise  # İptali bir üst katmana (handle_logic) bildir.
        finally:
            self.tts_task = None

    # <-- METOT GÜNCELLENDİ (llm_stream_complete eklendi) -->
    def run_llm_producer(self, text: str, lang: Literal["en", "tr"], queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, timings: dict):
        """
        (THREAD İÇİNDE ÇALIŞIR)
        LLM stream'ini tüketir, cümlelere böler ve kuyruğa atar.
        """
        buffer = ""
        # Cümle bölme regex'i (Virgülü eklemek isteğe bağlıdır, daha hızlı ama kesik TTS sağlar)
        sentence_pattern = re.compile(r'(?<=[.?!])\s+') 
        # sentence_pattern = re.compile(r'(?<=[.?!,])\s+') # Agresif bölme
        
        first_token = True

        try:
            for chunk in self.llm.stream_response(text, lang):
                
                if first_token:
                    timings['first_llm_token'] = time.monotonic() # LLM ilk token
                    first_token = False

                if "message" in chunk and "content" in chunk["message"]:
                    piece = chunk["message"]["content"]
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                    buffer += piece

                    sentences = sentence_pattern.split(buffer)

                    if len(sentences) > 1:
                        for i in range(len(sentences) - 1):
                            sentence_to_play = sentences[i].strip()
                            if sentence_to_play:
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(sentence_to_play), loop
                                ).result()

                        buffer = sentences[-1]

            final_piece = buffer.strip()
            if final_piece:
                asyncio.run_coroutine_threadsafe(
                    queue.put(final_piece), loop
                ).result()

        except Exception as e:
            print(f"\n[ERROR] LLM Producer thread'inde hata: {e}")
        finally:
            # LLM stream tamamlanma zamanı
            timings['llm_stream_complete'] = time.monotonic() # <-- YENİ
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    # <-- METOT GÜNCELLENDİ (Tüm 'say' çağrılarına 'timings' eklendi) -->
    async def _handle_logic(self, text: str, lang: Literal["en", "tr"], timings: dict):
        loop = asyncio.get_running_loop()
        # If we are waiting for YES/NO:
        if self.awaiting_confirmation and self.pending_station:
            # ... (Confirmation logic remains same)
            if self.agent.is_yes(text):
                ok, msg = await self.agent.send_navigation(self.pending_station)
                self.awaiting_confirmation = False
                station = self.pending_station
                self.pending_station = None
                if ok:
                    if lang == "en":
                        await self.say(f"Great. Guiding you to {station} now.",lang=lang,timings=timings)
                    else:
                        await self.say(f"Harika. Sizi {station} noktasına yönlendiriyorum.",lang=lang,timings=timings)
                else:
                    if lang == "en":
                        await self.say(f"Sorry, I couldn’t start navigation. {msg}",lang=lang,timings=timings)
                    else:
                        await self.say(f"Üzgünüm navigasyonu başlatamıyorum. {msg}",lang=lang,timings=timings)
                return
            elif self.agent.is_no(text):
                self.awaiting_confirmation = False
                self.pending_station = None
                if lang == "en":
                    await self.say("Okay. Let me know if you need anything else.",lang=lang,timings=timings)
                else:
                    await self.say("Tamamdır. Başka bir şeye ihtiyacın olursa söylemekten çekinme!",lang=lang,timings=timings)
                return
            else:
                if lang == "en":
                    await self.say("Please say yes to start or no to cancel.",lang=lang,timings=timings)
                else:
                    await self.say("Başlamak için evet, istemiyorsan hayır diyebilirsin.",lang=lang,timings=timings)
                return

        # No pending confirmation → classify intent
        
        # --- DEĞİŞİKLİK 1: INTENT CLASSIFIER'I AÇIKÇA ZAMANLA ---
        decision = {}
        try:
            decision = await loop.run_in_executor(
                self.llm_executor,
                self.agent.classify_intent,
                text
            )
        except Exception as e:
            print(f"[ERROR] Birleşik sınıflandırıcı (classify_intent) hatası: {e}")
            decision = {"request_type": "chat", "reason": "Sınıflandırma sırasında hata."}
        timings['intent_classifier_end'] = time.monotonic() # <-- YENİ ZAMAN DAMGASI
        request_type = decision.get("request_type")
        # --- DEĞİŞİKLİK 1 BİTTİ ---
        if request_type == "navigation":
            station = decision.get("target_station")
            if station and station in self.agent.station_names:
                # Navigasyon onayı iste
                self.awaiting_confirmation = True
                self.pending_station = station
                prompt = self.agent.build_confirmation_prompt(self.pending_station, lang)
                await self.say(prompt, lang, timings=timings)
            else:
                # LLM, var olmayan bir istasyon döndürürse
                print(f"[ERROR] LLM, geçersiz istasyon döndürdü: {station}")
                await self.say("Nereye gitmek istediğini anladım ama o istasyonu bulamadım.", lang, timings=timings)
        
        elif request_type == "iot":
            if not self.iot:
                print("[ERROR] IoT komutu algılandı ancak IoT servisi başlatılamadı.")
                await self.say("Akıllı cihaz komutunu anladım ancak kontrol servisine şu anda bağlı değilim.", lang, timings=timings)
                return

            target_code = decision.get("target_device_code")
            action = decision.get("action")
            value = decision.get("value") # (Bu, null olabilir)

            try:
                # <-- GÜNCELLENDİ: 'lang' parametresi eklendi -->
                ok, message = await loop.run_in_executor(
                    self.iot_executor,
                    self.agent.execute_iot_command, 
                    target_code,
                    action,
                    value,
                    lang  # <-- Dil parametresi buraya eklendi
                )
                
                # Kullanıcıya geri bildirim ver
                await self.say(message, lang, timings=timings)

            except Exception as e:
                print(f"[ERROR] IoT executor'da kritik hata: {e}")
                error_msg = "Akıllı cihazlara komut gönderirken beklenmedik bir hata oluştu."
                await self.say(error_msg, lang, timings=timings)

        else:
            # LLM Akışı ve Streaming TTS
            sys.stdout.write("[ASSISTANT] ")
            sys.stdout.flush()

            queue = asyncio.Queue()

            # --- DEĞİŞİKLİK 2: STREAMING LLM BAŞLANGICINI ZAMANLA ---
            timings['llm_stream_start'] = time.monotonic() # <-- YENİ ZAMAN DAMGASI
            # --- DEĞİŞİKLİK 2 BİTTİ ---
            
            producer_task = loop.run_in_executor(
                self.llm_executor,
                self.run_llm_producer,
                text,
                lang,
                queue,
                loop,
                timings
            )

            sentences_spoken = 0
            try:
                # ... (Kalan kod aynı) ...
                while True:
                    sentence = await queue.get()
                    if sentence is None:
                        break
                    
                    if sentences_spoken == 0 and 'first_sentence_received' not in timings:
                         timings['first_sentence_received'] = time.monotonic()

                    await self.say(sentence, lang, timings=timings)
                    sentences_spoken += 1
                print()
            except asyncio.CancelledError:
                print("\n[_handle_logic] Streaming TTS iptal edildi (Barge-in).")
                raise
            finally:
                if sentences_spoken == 0:
                    if lang == "en":
                        answer = "How can I help you?"
                    else:
                        answer = "Nasıl yardımcı olabilirim?"
                    try:
                        await self.say(answer, lang, timings=timings)
                    except asyncio.CancelledError:
                         print("[_handle_logic] Varsayılan yanıt sırasında iptal.")
                         raise

    # <-- METOT TAMAMEN YENİLENDİ -->
    def _print_timings(self, timings: dict, start_time: float):
        """Performans raporunu terminale yazdırır."""
        
        def get_time(key):
            # Zaman damgasını ve T0'dan farkını alır
            t = timings.get(key)
            return (t - start_time) if t else None

        # --- Tüm zaman farklarını hesapla ---
        d_stt_end = get_time('stt_end')
        
        # STT bittiyse devam et
        if d_stt_end is None:
            print("[T + ??????s] STT decode bitti (Hata)")
            return

        d_logic_start = get_time('logic_start')
        d_intent_end = get_time('intent_classifier_end')
        
        d_llm_stream_start = get_time('llm_stream_start')
        d_llm_first_token = get_time('first_llm_token')
        d_llm_complete = get_time('llm_stream_complete')
        
        d_tts_start = get_time('first_tts_start')
        d_tts_ready = get_time('first_tts_audio_ready')
        d_tts_play = get_time('first_audio_playback')
        d_total_end = get_time('total_process_end')

        # --- Raporu Temiz Bir Şekilde Bas ---
        print("\n--- PERFORMANS RAPORU (T=0: Konuşma Bitişi) ---")
        
        # 1. STT Aşaması
        # Bu, T=0'dan STT'nin bitişine kadar geçen saf süredir.
        print(f"[T + {d_stt_end:6.2f}s] 1. STT Decode Bitti (Süre: {d_stt_end:.2f}s)")

        # 2. Intent Aşaması
        if d_intent_end:
            # Bu, STT bittikten sonra Intent (non-streaming LLM) çağrısının
            # ne kadar sürdüğüdür.
            intent_duration = d_intent_end - d_logic_start
            print(f"[T + {d_intent_end:6.2f}s] 2. Intent Classifier Bitti (Süre: {intent_duration:.2f}s)")
        else:
            # Bu, bir 'evet/hayır' yanıtı veya navigasyon onayıysa
            print(f"[T + {d_logic_start:6.2f}s] 2. Logic (Confirmation) Başladı")

        # 3. LLM Yanıt Aşaması (Sadece 'else' durumunda çalışır)
        if d_llm_stream_start:
            print(f"[T + {d_llm_stream_start:6.2f}s] 3. Yanıt LLM Stream Başladı")
        
        if d_llm_first_token:
            # Bu, 'stream_start' ile ilk token'ın gelmesi arasındaki
            # saf 'Time-to-First-Token' gecikmesidir.
            ttft_duration = d_llm_first_token - (d_llm_stream_start or d_intent_end or d_logic_start)
            print(f"[T + {d_llm_first_token:6.2f}s]    -> LLM İlk Token Alındı (Gecikme: {ttft_duration:.2f}s)")
        
        if d_llm_complete:
            stream_duration = d_llm_complete - (d_llm_first_token or d_llm_stream_start or d_intent_end)
            print(f"[T + {d_llm_complete:6.2f}s]    -> LLM Stream Tamamlandı (Token Süresi: {stream_duration:.2f}s)")

        # 4. TTS Aşaması
        if d_tts_start:
             print(f"[T + {d_tts_start:6.2f}s] 4. İlk TTS Sentezi Başladı")
        
        if d_tts_ready:
            tts_synth_duration = d_tts_ready - d_tts_start
            print(f"[T + {d_tts_ready:6.2f}s]    -> İlk TTS Ses Üretildi (Sentez Süresi: {tts_synth_duration:.2f}s)")

        if d_tts_play:
            print(f"[T + {d_tts_play:6.2f}s] 5. İLK SES ÇALINDI (Time-to-First-Audio: {d_tts_play:.2f}s)")

        # 5. Toplam
        if d_total_end:
            print(f"[T + {d_total_end:6.2f}s] 6. TÜM SES ÇALMA BİTTİ (Toplam Süre: {d_total_end:.2f}s)")
        
        print("--- RAPOR SONU ---\n")

    # <-- METOT GÜNCELLENDİ (total_process_end eklendi) -->
    async def process_utterance(self, utter_pcm):
        """STT, LLM ve TTS'i birleştiren ana yanıt işleme görevi."""
        
        timings = {}  # Zamanlama sözlüğü
        start_time = time.monotonic()  # T0 (Başlangıç zamanı)
        
        try:
            # 1. STT
            text, lang = await asyncio.get_event_loop().run_in_executor(self.stt_executor, self.stt.stt, utter_pcm)
            timings['stt_end'] = time.monotonic()  # STT bitişi
            
            if not text:
                return

            print(f"\n[USER] {text}")
            if lang not in ["en", "tr"]:
                lang = "en"

            # 2. Logic / LLM / TTS
            timings['logic_start'] = time.monotonic()  # Logic başlama
            await self._handle_logic(text, lang, timings) # timings'i ilet
            
        except asyncio.CancelledError:
            print("[RESPONSE_TASK] Yanıt işleme görevi iptal edildi (Barge-in).")
            timings['cancelled'] = True  # Raporu basmamak için işaretle
            raise
        except Exception as e:
            print(f"[ERROR] Yanıt işleme sırasında beklenmedik hata: {e.__class__.__name__}: {e}")
            
        finally:
            # İptal edilmediyse toplam bitiş zamanını yakala
            if not timings.get('cancelled', False):
                timings['total_process_end'] = time.monotonic() # <-- YENİ

            # Sadece iptal edilmediyse (barge-in) ve STT bittiyse raporu yazdır.
            if not timings.get('cancelled', False) and 'stt_end' in timings:
                self._print_timings(timings, start_time)
                
            self.current_response_task = None


    async def run(self):
        await self.vad.start()
        print("Listening... (speak to the mic)")
        
        try:
            async for is_started, utter_pcm in self.vad.utterances():
                
                if is_started:
                    print("[BARGE-IN] Konuşma algılandı...")
                    if self.current_response_task and not self.current_response_task.done():
                        print("[BARGE-IN] Kullanıcı konuşmaya başladı. Önceki yanıt iptal ediliyor.")
                        self.current_response_task.cancel()
                    continue 
                
                if utter_pcm:
                    print("[BARGE-IN] Konuşma tamamlandı.")
                    if self.current_response_task and not self.current_response_task.done():
                        print("[WARNING] Önceki yanıt görevi hala çalışıyor. Yeni utterance göz ardı edildi.")
                        continue
                        
                    self.current_response_task = asyncio.create_task(self.process_utterance(utter_pcm))
                    
        except KeyboardInterrupt:
            print("[SHUTDOWN] Keyboard interrupt alındı.")
            pass
        
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Ana 'run' döngüsünde beklenmedik bir hata oluştu: {e.__class__.__name__}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print("[SHUTDOWN] Uygulama sonlanıyor. Kaynaklar temizleniyor...")
            
            await self.vad.stop()

            if self.current_response_task and not self.current_response_task.done():
                 self.current_response_task.cancel()
                 try:
                     await asyncio.wait_for(self.current_response_task, timeout=1.0)
                 except (asyncio.CancelledError, asyncio.TimeoutError):
                     pass
                     
            if self.tts_task and not self.tts_task.done():
                 self.tts_task.cancel()
                 try:
                     await asyncio.wait_for(self.tts_task, timeout=1.0)
                 except (asyncio.CancelledError, asyncio.TimeoutError):
                     pass

            self.stt_executor.shutdown(wait=False)
            self.llm_executor.shutdown(wait=False)
            self.iot_executor.shutdown(wait=False)
            self.tts.shutdown()