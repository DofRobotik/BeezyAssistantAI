from ddgs import DDGS
from typing import Literal


def search_web(query: str, max_results=4, lang: Literal["en", "tr"] = "tr") -> str:
    print(f"[SEARCH] '{query}' için internet taranıyor...")
    try:
        results = []
        with DDGS() as ddgs:
            # .text() metodu en hızlı ve genel sonuçları verir
            if lang == "tr":
                region = "turkey"
            else:
                region = "america"
            ddgs_gen = ddgs.text(query, max_results=max_results, region=region)
            if ddgs_gen:
                for r in ddgs_gen:
                    results.append(
                        f"- Title: {r.get('title')}\n  Snippet: {r.get('body')}"
                    )
            else:
                return "Arama sonucu bulunamadı."

        return "\n\n".join(results)
    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return "İnternet bağlantısı hatası veya arama servisi yanıt vermedi."
