from typing import Optional, List, Dict, Any, Tuple, Literal
import time
import httpx
import json

from utils import safe_json_find
from llm import LLM

# iot.py'den AmrLoungeClass'ı import et (Bu, cihaz listesini çekmek için gereklidir)
try:
    from .iot import AmrLoungeClass
except ImportError:
    print("Warning: .iot import failed, trying direct import 'iot'")
    from iot import AmrLoungeClass

# StationRouter sınıfı artık gerekli değil ve kaldırıldı.


class Agent:
    """
    Kullanıcı girdisini analiz etmek için BİRLEŞİK bir LLM kullanır ve
    'chat', 'navigate' veya 'iot' eylemlerinden birine karar verir.
    """

    def __init__(
        self,
        llm: LLM,
        stations: List[Dict[str, str]],
        iot_controller: AmrLoungeClass,
        robot_id: str = "amr-1",
        ros_nav_endpoint="http://10.10.190.14:8000/navigate",
    ):

        self.llm = llm
        self.stations = stations
        self.iot = iot_controller  # AmrLoungeClass örneği
        self.robot_id = robot_id
        self.ros_nav_endpoint = ros_nav_endpoint

        # --- LLM Prompt'ları için Varlık Listelerini ve IoT Haritalarını Hazırla ---
        self.station_names = [s["name"] for s in self.stations]
        self.station_prompt_list = "\n".join(
            [f"- {s['name']}: {s['property']}" for s in self.stations]
        )

        self.light_device_map = {}  # code -> {group, index}
        self.ac_device_map = {}  # code -> {index}

        all_iot_device_codes = []

        # Işık cihazlarını işle
        try:
            for group_index, devices in self.iot._AmrLoungeClass__lounge_place.items():
                for place_index, device in enumerate(devices):
                    code = device["code"]
                    all_iot_device_codes.append(code)
                    self.light_device_map[code] = {
                        "group": group_index,
                        "index": place_index,
                    }
        except Exception as e:
            print(f"[AGENT INIT] Işık cihazları haritası oluşturulamadı: {e}")

        # Klima cihazlarını işle
        # try:
        #     for aircon_index, device in enumerate(self.iot._AmrLoungeClass__aircon_list):
        #         code = device["code"]
        #         all_iot_device_codes.append(code)
        #         self.ac_device_map[code] = {
        #             "index": aircon_index
        #         }
        # except Exception as e:
        #     print(f"[AGENT INIT] Klima cihazları haritası oluşturulamadı: {e}")

        self.iot_device_prompt_list = ", ".join(all_iot_device_codes)
        # --- Hazırlık Bitti ---

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Sohbet geçmişini metne döker."""
        if not history:
            return "No previous conversation."

        formatted = []
        # Son 3 turu (6 mesaj) almak yeterli olabilir, token tasarrufu için
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    # router.py dosyasında _build_unified_prompt metodunu tamamen bu şekilde değiştirin:

    def _build_unified_prompt(self, text: str, history_str: str) -> str:
        # İstasyon açıklamalarını da ekleyelim ki "Yemek" denince "station_a" olduğunu anlasın
        return f"""
You are a decision-making router. Your job is to map the User Input to the correct Tool and Arguments.
Response must be a valid JSON object.

---
CONTEXT:
{history_str}

USER INPUT: "{text}"

---
AVAILABLE ASSETS (Strictly use these codes):

1. **VALID STATIONS** (For Navigation):
{self.station_prompt_list} 
(Use ONLY the station codes like 'station_a', 'station_b' etc.)

2. **VALID IOT DEVICES** (For Control):
[{self.iot_device_prompt_list}]
(Use ONLY these exact codes for 'target_device_code')

---
DECISION LOGIC:

1. **SEARCH (High Priority)**
   - Use this if the user asks for INFORMATION, FACTS, NEWS, DEFINITIONS, or about SPECIFIC ENTITIES (Companies, People, Products).
   - Example: "What is Hurricane?", "Hava nasıl?", "Dolar ne kadar?", "Tell me about yourself".
   - Output: {{"request_type": "search", "query": "optimized query"}}

2. **NAVIGATION**
   - Use this if the user wants to GO somewhere listed in "VALID STATIONS".
   - Example: "Go to the kitchen", "Take me to food court".
   - Output: {{"request_type": "navigation", "target_station": "station_code"}}

3. **IOT**
   - Use this if the user wants to CONTROL a device listed in "VALID IOT DEVICES".
   - Example: "Turn on the lounge light", "Işıkları kapat".
   - Output: {{"request_type": "iot", "target_device_code": "DEVICE_CODE", "action": "turn_on/off"}}

4. **CHAT**
   - Use this ONLY for simple Greetings ("Hello") or Confirmations ("Yes", "No").
   - DO NOT use Chat for questions like "What is X?". Use Search instead.

---
JSON SCHEMA:
{{
  "request_type": "search" | "navigation" | "iot" | "chat",
  "query": "string (optional, for search)",
  "target_station": "string (optional, exact station code)",
  "target_device_code": "string (optional, exact device code)",
  "action": "string (optional, 'turn_on' or 'turn_off')"
}}
"""

    def classify_intent(
        self, text: str, history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Hafıza destekli niyet analizi.
        """
        history_str = self._format_history(history)
        prompt = self._build_unified_prompt(text, history_str)

        # --- DEBUG BAŞLANGIÇ ---
        print(f"\n[ROUTER] Analiz ediliyor: '{text}'")
        # -----------------------

        # LLM'i JSON Modunda çağırıyoruz!
        raw_response = self.llm.complete(prompt, json_mode=True)

        # --- DEBUG DEVAM ---
        print(f"[ROUTER RAW OUTPUT]: {raw_response}")
        # Eğer burada mantıklı bir JSON görüp yine de search yapmıyorsa parser hatasıdır.
        # -------------------

        data = safe_json_find(raw_response)

        if not data:
            print("[ROUTER ERROR] JSON parse edilemedi, CHAT'e düşülüyor.")
            return {"request_type": "chat", "reason": "Parse error"}

        # Güvenlik kontrolü
        req_type = data.get("request_type")
        if req_type not in ["navigation", "iot", "chat", "search"]:
            print(f"[ROUTER WARN] Bilinmeyen tip: {req_type}, CHAT yapılıyor.")
            return {"request_type": "chat"}

        print(f"[ROUTER DECISION] >>> {req_type.upper()} <<<\n")
        return data

    def analyze_confirmation(self, text: str, context_action: str) -> str:
        """
        Kullanıcının onay durumunu analiz eder.
        Dönüş: 'yes', 'no' veya 'change_topic'
        """
        prompt = f"""
        TASK: Analyze the user's response to a confirmation request.
        
        CONTEXT: The robot asked: "Do you want to start {context_action}?"
        USER SAID: "{text}"
        
        INSTRUCTIONS:
        - If the user agrees (e.g., "yes", "sure", "ok", "evet", "tamam", "başla"): Output "yes"
        - If the user refuses (e.g., "no", "cancel", "stop", "hayır", "iptal", "vazgeç"): Output "no"
        - If the user ignores the question and talks about something else (e.g., "what is the weather?", "turn on lights", "merhaba"): Output "change_topic"
        
        OUTPUT ONLY ONE WORD: "yes", "no", or "change_topic"
        """
        response = self.llm.complete(prompt).strip().lower()

        if "yes" in response:
            return "yes"
        if "no" in response:
            return "no"
        return "change_topic"  # Varsayılan olarak konuyu değiştir

    def execute_iot_command(
        self, target_code: str, action: str, value: Any, lang: Literal["en", "tr"]
    ) -> Tuple[bool, str]:
        """
        (BLOKE EDİCİ) LLM'den gelen IoT komutunu çalıştırır.
        'iot_executor' içinde çalıştırılmalıdır.
        """
        print(
            f"[AGENT-IOT] Komut yürütülüyor: Cihaz={target_code}, Eylem={action}, Değer={value}"
        )

        try:
            # Durum 1: Cihaz bir IŞIK
            if target_code in self.light_device_map:
                device_info = self.light_device_map[target_code]
                group = device_info["group"]
                index = device_info["index"]

                if action == "turn_on":
                    # iot.py'den 'send_data_for_light_func' çağrısı
                    self.iot.send_data_for_light_func(
                        group, index, switch=True, dimming=150
                    )  # (Varsayılan parlaklık)
                    return True, (
                        f"{target_code} turned on."
                        if lang == "en"
                        else f"{target_code} açıldı."
                    )
                elif action == "turn_off":
                    # iot.py'den 'send_data_for_light_func' çağrısı
                    self.iot.send_data_for_light_func(
                        group, index, switch=False, dimming=0
                    )
                    return True, (
                        f"{target_code} turned off."
                        if lang == "en"
                        else f"{target_code} kapatıldı."
                    )
                else:
                    return False, (
                        f"'{action}' is an invalid action for lights."
                        if lang == "en"
                        else f"Işıklar için '{action}' geçersiz bir eylemdir."
                    )

            # # Durum 2: Cihaz bir KLİMA
            # elif target_code in self.ac_device_map:
            #     device_info = self.ac_device_map[target_code]
            #     index = device_info["index"]

            #     # iot.py'den 'send_data_for_aircon_func' çağrısı
            #     switch = True if action == "turn_on" else (False if action == "turn_off" else None)
            #     mode = value if action == "set_mode" else None
            #     temp = int(value) if action == "set_temperature" and value is not None else None
            #     fan = value if action == "set_fan_speed" else None

            #     if switch is None and action != "turn_off":
            #         switch = True

            #     self.iot.send_data_for_aircon_func(index, switch=switch, mode=mode, temperature=temp, fanspeed=fan)

            #     msg_en = f"{target_code} has been set (Action: {action}, Value: {value})."
            #     msg_tr = f"{target_code} ayarlandı (Eylem: {action}, Değer: {value})."
            #     return True, msg_en if lang == "en" else msg_tr

            else:
                return False, (
                    f"Device with code '{target_code}' not found."
                    if lang == "en"
                    else f"'{target_code}' kodlu cihaz sistemde bulunamadı."
                )

        except Exception as e:
            print(f"[AGENT-IOT-ERROR] Komut yürütülürken hata: {e}")
            msg_en = f"An error occurred while controlling the device: {str(e)}"
            msg_tr = f"Cihaz kontrolü sırasında bir hata oluştu: {str(e)}"
            return False, msg_en if lang == "en" else msg_tr

    def build_confirmation_prompt(self, station: str, lang: Literal["en", "tr"]) -> str:
        # ... (Bu kısım değişmedi) ...
        friendly_en = {
            "station_a": "the food court (station_a)",
            "station_b": "the restrooms (station_b)",
            "station_c": "the fun room (station_c)",
            "station_d": "the garment shop (station_d)",
            "station_e": "the tech shop (station_e)",
        }.get(station, f"{station}")
        friendly_tr = {
            "station_a": "Yemek bölümü (a istasyonu)",
            "station_b": "Lavabolar (b istasyonu)",
            "station_c": "Eğlence merkezi (c istasyonu)",
            "station_d": "Giyim mağazası (d istasyonu)",
            "station_e": "Teknoloji mağazası (e istasyonu)",
        }.get(station, f"{station}")
        if lang == "en":
            return f"{friendly_en} is nearby. Would you like me to guide you there?"
        else:
            return f"{friendly_tr} yakınlarda. Seni oraya götürmemi ister misin?"

    async def send_navigation(self, station: str) -> Tuple[bool, str]:
        payload = {"station": station, "source": self.robot_id, "ts": int(time.time())}
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.post(str(self.ros_nav_endpoint), json=payload)
            if r.status_code // 100 == 2:
                return True, "Navigation started."
            return False, f"ROS bridge error: {r.status_code} {r.text[:120]}"
        except Exception as e:
            print("[PAYLOAD] ", payload)
            return False, f"HTTP error: {e}"

    @staticmethod
    def is_yes(text: str) -> bool:
        t = text.strip().lower()
        return any(
            x in t
            for x in [
                "yes",
                "yeah",
                "yep",
                "please do",
                "ok",
                "sure",
                "confirm",
                "go ahead",
                "start",
                "guide me",
                "let's go",
                "let us go",
                "affirmative",
                "evet",
                "aynen",
                "doğru",
            ]
        )

    @staticmethod
    def is_no(text: str) -> bool:
        t = text.strip().lower()
        return any(
            x in t
            for x in [
                "no",
                "nope",
                "not now",
                "cancel",
                "stop",
                "don't",
                "do not",
                "negative",
                "hayır",
                "yok",
                "istemiyorum",
            ]
        )
