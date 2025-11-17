import ollama
from typing import Literal, List, Dict


class LLM:
    def __init__(self, model="qwen3:8b"):
        self.model = model

    def _build_system_message(self, lang: Literal["en", "tr"]) -> Dict[str, str]:
        content = (
            "Sen yardımcı ve zeki bir robotsun (Beezy). "
            "Sana bir 'Arama Sonucu' veya 'Bağlam' verilirse, cevabını ÖNCELİKLE ona dayandır. "
            "Eğer bağlam verilmezse veya yetersizse, kendi genel kültürünü kullanarak en iyi cevabı ver. "
            "Bilmediğin ve emin olmadığın konularda dürüstçe 'Bilmiyorum' de ama hemen pes etme. "
            "Cevapların kısa, net ve konuşma diline uygun olsun."
        )
        if lang == "en":
            content = (
                "You are a helpful and intelligent robot (Beezy). "
                "If 'Search Results' or 'Context' are provided, base your answer PRIMARILY on them. "
                "If context is missing or insufficient, use your own general knowledge to answer helpfully. "
                "Admit when you don't know something, but don't give up easily. "
                "Keep answers concise and conversational."
            )
        return {"role": "system", "content": content}

    def stream_chat(self, history: List[Dict[str, str]], lang: Literal["en", "tr"]):
        """
        Geçmişi (history) dikkate alarak streaming yanıt üretir.
        history formatı: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        # Sistem mesajını en başa ekle
        system_msg = self._build_system_message(lang)
        messages = [system_msg] + history

        return ollama.chat(
            model=self.model, messages=messages, stream=True, keep_alive=-1
        )

    def complete(
        self, prompt: str, temperature: float = 0.0, json_mode: bool = False
    ) -> str:
        # Options ayarları
        options = {
            "temperature": temperature,
            "num_ctx": 4096,  # Context penceresini geniş tutalım
        }

        # Eğer json_mode True ise format='json' gönderilir
        fmt = "json" if json_mode else None

        r = ollama.generate(
            model=self.model,
            prompt=prompt,
            format=fmt,  # <--- İŞTE SİHİRLİ DEĞİŞİKLİK
            stream=False,
            options=options,
            keep_alive=-1,
        )
        return r.get("response", "").strip()
