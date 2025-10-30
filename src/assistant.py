import sys
import asyncio
import re
import time
from typing import Optional, Literal,Dict,Callable
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
    def __init__(self,callbacks: Optional[Dict[str, Callable]] = None):
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

        # --- YENİ ---
        # UI callback'lerini ve çalışma durumunu sakla
        self.callbacks = callbacks if callbacks else {}
        self.running = False

    def _trigger_callback(self, name: str, *args):
        if name in self.callbacks:
            try:
                self.callbacks[name](*args)
            except Exception as e:
                print(f"[CALLBACK-ERROR] '{name}' callback'i tetiklenirken hata: {e}")

    # --- GÜNCELLENMİŞ METOT ---
    async def say(self, text: str, lang, timings: dict):
        """
        Asistanın konuşmasını başlatır (SADECE TTS).
        UI callback'i buradan kaldırıldı.
        """
        
        # Sadece ilk çağrıda 'TTS Sentez Başlangıcı' zaman damgasını al
        if 'first_tts_start' not in timings:
            timings['first_tts_start'] = time.monotonic()
            
        # UI tetikleyicisi buradan KALDIRILDI.
        
        self.tts_task = asyncio.create_task(self.tts.tts_and_play(text, lang, timings))
        try:
            await self.tts_task
        except asyncio.CancelledError:
            print("[ASSISTANT] Konuşma iptal edildi (Barge-in).")
            raise  # İptali bir üst katmana (handle_logic) bildir.
        finally:
            self.tts_task = None

    def run_llm_producer(self, text: str, lang: Literal["en", "tr"], queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, timings: dict):
        """
        (THREAD İÇİNDE ÇALIŞIR)
        LLM stream'ini tüketir, 'chunk' (UI için) ve 'sentence' (TTS için) 
        olarak kuyruğa atar.
        """
        buffer = ""
        sentence_pattern = re.compile(r'(?<=[.?!])\s+') 
        first_token = True

        def q_put(item):
            """Thread-safe kuyruğa ekleme yardımcısı."""
            asyncio.run_coroutine_threadsafe(queue.put(item), loop) # .result() kullanma, ana döngüyü bloklar

        try:
            for chunk in self.llm.stream_response(text, lang):
                
                if first_token:
                    timings['first_llm_token'] = time.monotonic() 
                    first_token = False

                if "message" in chunk and "content" in chunk["message"]:
                    piece = chunk["message"]["content"]
                    
                    # --- YENİ: UI'a anında 'chunk' gönder ---
                    q_put(("chunk", piece))
                    
                    buffer += piece
                    sentences = sentence_pattern.split(buffer)

                    if len(sentences) > 1:
                        for i in range(len(sentences) - 1):
                            sentence_to_play = sentences[i].strip()
                            if sentence_to_play:
                                # --- YENİ: TTS için 'sentence' gönder ---
                                q_put(("sentence", sentence_to_play))

                        buffer = sentences[-1]

        except Exception as e:
            print(f"\n[ERROR] LLM Producer thread'inde hata: {e}")
            q_put(("error", str(e))) # Hata durumunu da bildir
        finally:
            # LLM stream tamamlanma zamanı
            timings['llm_stream_complete'] = time.monotonic() 
            
            # Kalan son cümleyi de TTS için gönder
            final_piece = buffer.strip()
            if final_piece:
                q_put(("sentence", final_piece))
            
            # --- YENİ: Bitiş sinyalini gönder ---
            q_put(("end", None))

    # <-- METOT GÜNCELLENDİ (Tüm 'say' çağrılarına 'timings' eklendi) -->
    async def _handle_logic(self, text: str, lang: Literal["en", "tr"], timings: dict):
        loop = asyncio.get_running_loop()
        # If we are waiting for YES/NO:
        if self.awaiting_confirmation and self.pending_station:
            async def say_and_show(msg: str):
                self._trigger_callback("on_response_start") # Baloncuk yarat
                self._trigger_callback("on_response_chunk", msg) # Doldur
                self._trigger_callback("on_response_end") # Bitir
                await self.say(msg, lang, timings=timings) # Seslendir
            
            # ... (Confirmation logic remains same)
            if self.agent.is_yes(text):
                ok, msg = await self.agent.send_navigation(self.pending_station)
                self.awaiting_confirmation = False
                station = self.pending_station
                self.pending_station = None
                if ok:
                    if lang == "en":
                        await say_and_show(f"Great. Guiding you to {station} now.")
                    else:
                        await say_and_show(f"Harika. Sizi {station} noktasına yönlendiriyorum.")
                else:
                    if lang == "en":
                        await say_and_show(f"Sorry, I couldn’t start navigation. {msg}")
                    else:
                        await say_and_show(f"Üzgünüm navigasyonu başlatamıyorum. {msg}")
                return
            elif self.agent.is_no(text):
                self.awaiting_confirmation = False
                self.pending_station = None
                if lang == "en":
                    await say_and_show("Okay. Let me know if you need anything else.")
                else:
                    await say_and_show("Tamamdır. Başka bir şeye ihtiyacın olursa söylemekten çekinme!")
                return
            else:
                if lang == "en":
                    await say_and_show("Please say yes to start or no to cancel.")
                else:
                    await say_and_show("Başlamak için evet, istemiyorsan hayır diyebilirsin.")
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

        async def say_and_show(msg: str):
            self._trigger_callback("on_response_start")
            self._trigger_callback("on_response_chunk", msg)
            self._trigger_callback("on_response_end")
            await self.say(msg, lang, timings=timings)

        # --- DEĞİŞİKLİK 1 BİTTİ ---
        if request_type == "navigation":
            station = decision.get("target_station")
            if station and station in self.agent.station_names:
                # Navigasyon onayı iste
                self.awaiting_confirmation = True
                self.pending_station = station
                prompt = self.agent.build_confirmation_prompt(self.pending_station, lang)
                await say_and_show(prompt)
            else:
                # LLM, var olmayan bir istasyon döndürürse
                print(f"[ERROR] LLM, geçersiz istasyon döndürdü: {station}")
                await say_and_show("Nereye gitmek istediğini anladım ama o istasyonu bulamadım.")
        
        elif request_type == "iot":
            if not self.iot:
                print("[ERROR] IoT komutu algılandı ancak IoT servisi başlatılamadı.")
                await say_and_show("Akıllı cihaz komutunu anladım ancak kontrol servisine şu anda bağlı değilim.")
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
                await say_and_show(message)

            except Exception as e:
                print(f"[ERROR] IoT executor'da kritik hata: {e}")
                error_msg = "Akıllı cihazlara komut gönderirken beklenmedik bir hata oluştu."
                await say_and_show(error_msg)

        else:
            # --- YENİ: STREAMING CHAT MANTIĞI ---
            timings['llm_stream_start'] = time.monotonic() 
            
            queue = asyncio.Queue()
            producer_task = loop.run_in_executor(
                self.llm_executor,
                self.run_llm_producer,
                text, lang, queue, loop, timings
            )

            first_chunk = True
            try:
                while True:
                    item = await queue.get()
                    if not item: 
                        print("[WARN] Kuyruktan boş öğe alındı.")
                        continue
                    
                    msg_type, data = item
                    
                    if msg_type == "chunk":
                        if first_chunk:
                            # İlk 'chunk' geldiğinde UI'ı başlat
                            self._trigger_callback("on_response_start")
                            sys.stdout.write("[ASSISTANT] ")
                            sys.stdout.flush()
                            first_chunk = False
                        
                        # UI'a ve terminale chunk'ı yaz
                        self._trigger_callback("on_response_chunk", data)
                        sys.stdout.write(data)
                        sys.stdout.flush()
                    
                    elif msg_type == "sentence":
                        # TTS için tam cümleyi seslendir
                        if data:
                            await self.say(data, lang, timings=timings)
                    
                    elif msg_type == "end":
                        self._trigger_callback("on_response_end")
                        print() # Terminalde yeni satıra geç
                        break
                    
                    elif msg_type == "error":
                        print(f"[ERROR] Stream hatası alındı: {data}")
                        if first_chunk: # Hata oluştu ama hiç konuşulmadı
                           await say_and_show("Üzgünüm, bir hata oluştu." if lang == "tr" else "Sorry, an error occurred.")
                        break

            except asyncio.CancelledError:
                print("\n[_handle_logic] Streaming TTS iptal edildi (Barge-in).")
                producer_task.cancel() # Thread'i durdurmayı dene (doğrudan etki etmez ama executor'a sinyal verir)
                raise
            finally:
                # Hiçbir chunk gelmediyse (örn. LLM boş yanıt verdi)
                if first_chunk:
                    print("[_handle_logic] Hiçbir LLM chunk'ı alınmadı.")
                    await say_and_show("Nasıl yardımcı olabilirim?" if lang == "tr" else "How can I help you?")

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

            # --- YENİ ---
            # Arayüze "İşleniyor" sinyalini ve STT metnini gönder
            self._trigger_callback("on_processing_started", text)

            # 2. Logic / LLM / TTS
            timings['logic_start'] = time.monotonic()  # Logic başlama
            await self._handle_logic(text, lang, timings) # timings'i ilet
            
        except asyncio.CancelledError:
            print("[RESPONSE_TASK] Yanıt işleme görevi iptal edildi (Barge-in).")
            timings['cancelled'] = True  # Raporu basmamak için işaretle
            raise
        except Exception as e:
            print(f"[ERROR] Yanıt işleme sırasında beklenmedik hata: {e.__class__.__name__}: {e}")
            # Arayüze hata sinyali gönder
            self._trigger_callback("on_error", f"Yanıt işlenirken hata: {e}")
            
        finally:
            # İptal edilmediyse toplam bitiş zamanını yakala
            if not timings.get('cancelled', False):
                timings['total_process_end'] = time.monotonic() # <-- YENİ

            # Sadece iptal edilmediyse (barge-in) ve STT bittiyse raporu yazdır.
            if not timings.get('cancelled', False) and 'stt_end' in timings:
                self._print_timings(timings, start_time)
            
            # --- YENİ ---
            # Görev bittiğinde (iptal edilmediyse) arayüze "Hazır" sinyali gönder
            if not timings.get('cancelled', False):
                self._trigger_callback("on_ready")
                
            self.current_response_task = None

    # --- YENİ ---
    # UI'dan çağırmak için stop metodu
    async def stop(self):
        """Asistanı ve VAD'ı güvenli bir şekilde durdurur."""
        print("[SHUTDOWN] Stop komutu alındı.")
        if self.running:
            await self.vad.stop() # Bu, 'run' metodundaki 'async for' döngüsünü kıracaktır
        self.running = False

    async def run(self):
        self.running = True # Çalışma bayrağını ayarla
        await self.vad.start()
        print("Listening... (speak to the mic)")
        # --- YENİ ---
        # Başlangıçta arayüze "Hazır" sinyali gönder
        self._trigger_callback("on_ready")

        try:
            async for is_started, utter_pcm in self.vad.utterances():
                
                # --- YENİ ---
                # UI'ı durdurmak için 'self.running' bayrağını kontrol et
                if not self.running:
                    print("[RUN] 'self.running' False olarak ayarlandı, döngüden çıkılıyor.")
                    break

                if is_started:
                    print("[BARGE-IN] Konuşma algılandı...")
                    self._trigger_callback("on_listening_started")

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
            # --- YENİ ---
            self._trigger_callback("on_error", f"Kritik hata: {e}")

        finally:
            print("[SHUTDOWN] Uygulama sonlanıyor. Kaynaklar temizleniyor...")
            
            # VAD'ın zaten durdurulmuş olması gerekir, ancak tekrar kontrol et
            if self.vad.running:
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
            
            self.running = False # Tamamen durduğunu işaretle
            print("[SHUTDOWN] Temizleme tamamlandı.")