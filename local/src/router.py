from typing import Optional, List, Dict, Any, Tuple, Literal
import time
import httpx
import json

from .utils import safe_json_find
from .llm import LLM

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

    def _build_unified_prompt(self, text: str) -> str:
        """
        Navigasyon, IoT ve Chat'i kapsayan birleşik LLM prompt'unu oluşturur.
        """
        return f"""
You are the decision-making core for a robot assistant. Your task is to analyze the user's input and classify it into one of three categories: "navigation", "iot", or "chat".

User Request: "{text}"

---
CHECKLIST AND RULES:

1. NAVIGATION (navigation)
Use this if the user wants to go to one of the following stations.
Stations:
{self.station_prompt_list}
Valid Station Codes: {self.station_names}

2. SMART DEVICE (iot)
Use this if the user wants to control (turn on/off, set) one of the following devices.
Device Codes: {self.iot_device_prompt_list}
Valid Actions (action):
 - For lights: "turn_on", "turn_off"

3. CHAT (chat)
Use this if the request is not related to navigation or IoT (e.g., greetings, asking for the weather, general questions).

---
JSON OUTPUT FORMAT:
You must respond ONLY with a SINGLE JSON object that matches one of the following formats.

For Navigation:
{{
  "request_type": "navigation",
  "target_station": "<One of the station codes, e.g., 'station_a'>",
  "reason": "<short justification>"
}}

For Smart Device:
{{
  "request_type": "iot",
  "target_device_code": "<One of the device codes, e.g., 'ORTAK_ALAN_TABELA'>",
  "action": "<One of the valid actions, e.g., 'turn_off' or 'turn on'>",
  "reason": "<short justification>"
}}

For Chat:
{{
  "request_type": "chat",
  "reason": "<short justification, e.g., 'general chat', 'greeting'>"
}}
"""

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        (BLOKE EDİCİ) Birleşik LLM'i çağırır ve sonucu JSON olarak döndürür.
        Bu metot 'run_in_executor' içinde çalıştırılmalıdır.
        """
        prompt = self._build_unified_prompt(text)

        # LLM'i çağır (Bu non-streaming, bloke edici bir çağrıdır)
        raw_response = self.llm.complete(prompt)

        # Yanıtı JSON olarak bul ve ayrıştır
        data = safe_json_find(raw_response)

        if not data:
            print(
                f"[AGENT-ERROR] LLM'den geçerli JSON alınamadı. Yanıt: {raw_response}"
            )
            return {"request_type": "chat", "reason": "LLM yanıtı ayrıştırılamadı"}

        # JSON'un geçerli bir 'request_type' içerdiğinden emin ol
        if data.get("request_type") not in ["navigation", "iot", "chat"]:
            print(
                f"[AGENT-WARN] LLM'den bilinmeyen request_type alındı: {data.get('request_type')}"
            )
            return {"request_type": "chat", "reason": "Bilinmeyen request_type"}

        print(f"[AGENT-CLASSIFY] Karar: {json.dumps(data)}")
        return data

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
