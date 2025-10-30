import numpy as np
import re
import json
from typing import Optional,Dict,Any

def pcm16_to_bytes(audio_np):
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return (audio_np * 32767).astype(np.int16).tobytes()

def bytes_to_pcm16(b):
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0

def safe_json_find(text:str) -> Optional[Dict[str,Any]]:
    """
    Extract the first {...} JSON object and parse it safely.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None