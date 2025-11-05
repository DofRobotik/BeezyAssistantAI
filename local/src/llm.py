import ollama
from typing import Literal


class LLM:
    def __init__(self, model="qwen3:8b"):
        self.model = model

    def _build_system_prompt(self, lang: Literal["en", "tr"]) -> str:
        """Dile göre sistem prompt'u oluşturur."""
        if lang == "tr":
            return (
                "Sen DOF Robotics'in Otonom Mobil Robot asistanısın. "
                "Kısa ve öz, yardımcı olacak şekilde yanıt ver. Yalnızca Türkçe yanıt ver. "
                "Markdown formatı kullanma."
            )
        else:  # Varsayılan olarak İngilizce
            return (
                "You are DOF Robotics' Autonomous Mobile Robot assistant. "
                "Be concise, helpful, and respond in English only. "
                "Avoid Markdown formatting."
            )

    def stream_response(self, user_text: str, lang: Literal["en", "tr"]):
        """LLM'den streaming yanıt alır (dile duyarlı)."""
        system_prompt = self._build_system_prompt(lang)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        return ollama.chat(model=self.model, messages=messages, stream=True, think=False, keep_alive=-1)

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        # Non-streaming single-shot completion (for control JSON, classification, etc.)
        # NOT: Agent (classify_intent) şu anda _handle_logic'ten 'lang' almıyor.
        # Bu yüzden bu metot varsayılan (İngilizce) sistem prompt'u olmadan çalışır.
        # Bu, agent'ın (sınıflandırma vs.) prompt'larının kendi içinde
        # dile duyarlı olmadığını varsayar. Bu genellikle sorun değildir.
        options = {"temperature": temperature}
        r = ollama.generate(model=self.model, prompt=prompt, think=False, options=options, keep_alive=-1)
        return r.get("response", "").strip()