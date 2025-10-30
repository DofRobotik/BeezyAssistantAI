import numpy as np
import faster_whisper

class STTPipe:
    def __init__(self,model_path:str = "faster-whisper-base"):
        self.model = faster_whisper.WhisperModel(model_path, device="cuda", local_files_only=True,compute_type="float16")

    def stt(self, audio_data: bytes):
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float16) / 32768.0
        text = ""
        segments, info = self.model.transcribe(
            audio_np,
            beam_size=10,
            no_speech_threshold=0.8,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.0,
            suppress_blank=True,
            initial_prompt=None,
            suppress_tokens=[-1] if True else None,
            condition_on_previous_text=True,
            temperature=0,
            task="transcribe",
            multilingual=True,
            log_progress=True,
            language_detection_threshold=0.80

        )
        for seg in segments:
            text += seg.text
        return text.strip(), info.language