#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import json
import random

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


# http://192.168.1.114:3001/places/
class AmrLoungeClass(object):
    def __init__(self, service_url="localhost", port=3001):
        self.__service_url = self.__generate_url(service_url, port)
        self.__location = "places"
        self.__aircon_key = "k"
        print("Current URL = {}".format(self.__service_url))

        self.__lounge_place = {
            "a": [
                {"code": "LOUNGE_GENEL", "switch": True, "dimming": None},
                {"code": "LOUNGE_GIRIS", "switch": True, "dimming": None},
                {"code": "LOUNGE_MERDIVEN_USTU", "switch": True, "dimming": None},
                {"code": "ORTAK_ALAN", "switch": True, "dimming": None},
                {"code": "BITKI_LED", "switch": True, "dimming": 0},
                {"code": "ORTAK_ALAN_TABELA", "switch": True, "dimming": 0},
                {"code": "DUVAR_LED", "switch": True, "dimming": 0},
                {"code": "DIS_TABELA", "switch": True, "dimming": 0},
            ],
            "b": [
                {"code": "GORUSME_ODA_GENEL", "switch": True, "dimming": None},
                {"code": "GORUSME_ODA1", "switch": True, "dimming": None},
                {"code": "GORUSME_ODA2", "switch": True, "dimming": None},
                {"code": "GORUSME_ODA3", "switch": True, "dimming": None},
                {"code": "GORUSME_ODA1_LED", "switch": True, "dimming": 0},
                {"code": "GORUSME_ODA2_LED", "switch": True, "dimming": 0},
                {"code": "GORUSME_ODA3_LED", "switch": True, "dimming": 0},
            ],
            "c": [
                {"code": "AMFI_GENEL", "switch": True, "dimming": None},
                {"code": "AMFI_SPOT", "switch": True, "dimming": None},
                {"code": "AMFI_BASAMAK", "switch": True, "dimming": 0},
                {"code": "AMFI_MOR_LED", "switch": True, "dimming": 0},
                {"code": "AMFI_TABELA", "switch": True, "dimming": 0},
            ],
            "d": [
                {"code": "MUTFAK_GENEL", "switch": True, "dimming": None},
                {"code": "MUTFAK_SARKIT", "switch": True, "dimming": None},
                {"code": "MUTFAK_SPOT", "switch": True, "dimming": None},
                {"code": "MUTFAK_TABELA", "switch": True, "dimming": 0},
            ],
            "e": [
                {"code": "UST_TOPLANTI_ODA_GENEL", "switch": True, "dimming": None},
                {"code": "UST_TOPLANTI_ODA_KENAR", "switch": True, "dimming": None},
                {"code": "UST_TOPLANTI_ODA_ORTA", "switch": True, "dimming": None},
                {"code": "UST_TOPLANTI_ODA_SARKIT", "switch": True, "dimming": 0},
            ],
            "f": [
                {"code": "ALT_TOPLANTI_ODA_GENEL", "switch": True, "dimming": None},
                {"code": "ALT_TOPLANTI_ODA", "switch": True, "dimming": None},
                {"code": "ALT_TOPLANTI_ODA_BASAMAK", "switch": True, "dimming": 0},
                {"code": "ALT_TOPLANTI_ODA_LED", "switch": True, "dimming": 0},
            ],
            "g": [
                {"code": "WC_GENEL", "switch": True, "dimming": None},
                {"code": "WC_GENEL_KORIDOR", "switch": True, "dimming": None},
                {"code": "WC_ENGELLI", "switch": True, "dimming": 0},
                {"code": "WC_KADIN", "switch": True, "dimming": 0},
                {"code": "WC_ERKEK", "switch": True, "dimming": 0},
            ],
        }

        self.__group_name = {
            "a": "Genel",
            "b": "Gorusme Oda",
            "c": "Amfi",
            "d": "Mutfak",
            "e": "Ust Toplanti Oda",
            "f": "Alt Toplanti Oda",
            "g": "Tuvalet",
            "h": "Merkez",
            "k": "Klima",
        }

        self.__group_index = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
            "k": 8,
        }

        """
            "code": "KLIMA_GENEL",      # İsim
            "switch": True,             # Klima On / Off
            "mode": None,               # Yaz/Kış Mod geçişi (0=cool / 1=heat)
            "fanspeed": 0,              # Fan hızı (0 = auto / 1 = fan - 1 / 2 = fan - 2 / 3 = fan - 3)
            "temperature": None         # Set point temperature control (ortam sıcaklığı belirtmek için kullanılır)
        """
        self.__aircon_list = [
            {
                "code": "KLIMA_GENEL",
                "switch": True,
                "mode": 0,
                "fanspeed": 0,
                "temperature": 25,
            },
            {
                "code": "KLIMA_GIRIS",
                "switch": True,
                "mode": 0,
                "fanspeed": 0,
                "temperature": 25,
            },
            {
                "code": "KLIMA_AMFI",
                "switch": True,
                "mode": 0,
                "fanspeed": 0,
                "temperature": 25,
            },
        ]

    """
    Verilen grup ve konum indexine göre grup adı ve o konuma ait cihaz verisini döner.

    Args:
        group_index (int): Grup listesindeki index.
        place_index (int): Grup içerisindeki yer listesindeki index.

    Returns:
        tuple: (group_name, place_value)
               - group_name (str): İlgili grubun adı.
               - place_value (dict): İlgili konuma ait cihaz verisi.
    
    Returns (on error):
        tuple: (None, None) Eğer hata oluşursa None değerleri döner.

    Notes:
        - Eğer verilen grup veya konum indexi geçerli değilse, hata yakalanır ve None döner.
    """

    def get_group_and_place_func(self, group_index, place_index):
        try:
            group_name = self.__group_name[group_index]
            place_value = self.__lounge_place[group_index][place_index]

            return group_name, place_value

        except Exception as err:
            print("get_group_and_place_func Error = {}".format(err))
            return None, None

    """
    Belirtilen bir aydınlatma cihazı için verileri hazırlayıp sunucuya gönderir.

    Args:
        group_index (int): Hangi grup içerisindeki aydınlatma cihazına işlem yapılacağını belirten index.
        place_index (int): Hangi konumdaki aydınlatma cihazına işlem yapılacağını belirten index.
        switch (bool): Aydınlatmayı açma/kapama durumu (True: Açık, False: Kapalı).
        dimming (int, optional): Aydınlatmanın dimming değeri (0-255 arası). Eğer None verilirse mevcut dimming değeri kullanılır.

    Returns:
        requests.Response or None: İstek başarılı olursa sunucudan dönen yanıt döner, aksi takdirde None döner.

    Notes:
        - Eğer dimming değeri None verilirse, cihazın mevcut dimming değeri kullanılır.
        - Eğer switch False olarak ayarlanmışsa, dimming değeri 0 olarak güncellenir (ışık kapatılır).
    """

    def get_all_devices(self):
        """Returns a dict mapping device_code to its metadata for LLM context."""
        devices = {}
        # Lights
        for grp, items in self.__lounge_place.items():
            for idx, item in enumerate(items):
                devices[item["code"]] = {"type": "light", "group": grp, "index": idx}
        # Aircon (Optional implementation)
        for idx, item in enumerate(self.__aircon_list):
            devices[item["code"]] = {"type": "aircon", "index": idx}
        return devices

    def send_data_for_light_func(self, group_index, place_index, switch, dimming=None):
        try:
            _, place_value = self.get_group_and_place_func(group_index, place_index)

            if dimming is None:
                dimming_value = place_value["dimming"]

            else:
                dimming_value = dimming

            # NOTE: Kontrole gore eklendi
            if not switch:
                dimming_value = 0

            data_to_send = self.__generate_data_func(
                group_index, place_value["code"], switch, dimming_value
            )  # place_value["dimming"]
            print("Data Set = {}".format(data_to_send))

            # TODO: Acilacak, test asamasinda
            return self.__send_request(data_to_send)
            # return True

        except Exception as err:
            print("Send Data Error = {}".format(err))
            return None

    """
    Genel bir cihaz için gönderilecek veriyi oluşturur.
    
    Args:
        group (int/str): Cihazın ait olduğu grup. String ya da int olabilir. Eğer None ise grup değeri rastgele atanabilir.
        code_name (str): Cihazın kod adı (örn: "ORTAK_ALAN_TABELA").
        switch (bool): Cihazın açma/kapama durumu (True: Açık, False: Kapalı).
        dimming (int, optional): Cihazın dimming değeri (0-255 arası). Eğer None ise rastgele bir değer atanır.
        name (str, optional): Cihazın adı. Eğer None ise code_name'den otomatik olarak oluşturulur.
    
    Returns:
        dict: Cihaza gönderilecek veriyi içeren sözlük.
    
    Notes:
        - "name" alanı sadece görsel amaçlıdır, gerekirse kaldırılabilir.
        - "groupId" alanı isteğe bağlıdır, string veya int olabilir.
        - Eğer dimming değeri verilmemişse rastgele 1-255 arasında bir değer atanır.
    """

    def __generate_data_func(self, group, code_name, switch, dimming=None, name=None):
        try:
            if name is None:
                name = str(code_name).capitalize().title().replace("_", " ")

            if dimming == None:
                dimming = random.randint(1, 255)
                # dimming = 1

            if group == None or isinstance(group, str):
                # group = random.randint(1, 20)
                group = self.__group_index[group]  # + 5   # Shift value

            data = {
                "code": code_name,  # String            (ORTAK_ALAN_TABELA)
                "name": name,  # String            (Ortak Alan Tabela)
                # NOTE: "name" attribute'u Sadece gorunus amacli, istenirse kaldirilabilir
                "switch": switch,  # Bool              (True / False)
                "dimming": dimming,  # Int               (0 - 255)
                "groupId": group,
            }  # String / Int      ("B" / 1)
            # NOTE: Group ID istege baglidir, string yada int olabilir

            return data

        except Exception as err:
            print("Generate Data Func Error = {}".format(err))
            return None

    """
    Klima cihazı için verileri hazırlayıp sunucuya gönderir.

    Args:
        aircon_index (int): Hangi klima cihazına işlem yapılacağını belirten index.
        switch (bool): Klimayı açma/kapama durumu (True: Açık, False: Kapalı).
        mode (str, optional): Klimanın çalışma modu (örn: "cool", "heat"). Eğer None verilirse mevcut mod kullanılır.
        fanspeed (str/int, optional): Klimanın fan hızı. Eğer None verilirse mevcut fan hızı kullanılır.
        temperature (int/float, optional): Klimanın ayarlanan sıcaklığı. Eğer None verilirse mevcut sıcaklık kullanılır.

    Returns:
        bool or None: Eğer veri başarıyla oluşturulursa True döner, hata durumunda None döner.

    Notes:
        - Mode, fanspeed ve temperature parametreleri isteğe bağlıdır. Eğer None verilirse cihazın o anki değeri kullanılır.
    """

    def send_data_for_aircon_func(
        self, aircon_index, switch, mode=None, fanspeed=None, temperature=None
    ):
        try:
            aircon_value = self.__aircon_list[aircon_index]

            if mode is None:
                mode = aircon_value["mode"]

            else:
                self.__aircon_list[aircon_index]["mode"] = mode

            if fanspeed is None:
                fanspeed = aircon_value["fanspeed"]

            else:
                self.__aircon_list[aircon_index]["fanspeed"] = fanspeed

            if temperature is None:
                temperature = aircon_value["temperature"]

            else:
                self.__aircon_list[aircon_index]["temperature"] = temperature

            """
                {
                    "code": "KLIMA_GENEL",
                    "switch": True,
                    "mode": 0,
                    "fanspeed": 0,
                    "temperature": 25
                }
            """

            data_to_send = self.__generate_data_for_aircon_func(
                aircon_value["code"], switch, mode, fanspeed, temperature
            )
            print("\n\nData Set = {}\n\n".format(data_to_send))

            return self.__send_request(data_to_send)
            return True

        except Exception as err:
            print("Send Data Error = {}".format(err))
            return None

    """
    Klima cihazı için gerekli olan veriyi oluşturur.
    
    Args:
        code_name (str): Klimanın kod adı (örn: "KLIMA_GIRIS").
        switch (bool): Klimayı açıp kapatma durumu (True: Açık, False: Kapalı).
        mode (str): Klimanın çalışma modu (örn: "cool", "heat").
        fanspeed (str/int): Klimanın fan hızı (örn: "low", "medium", "high" veya sayısal bir değer).
        temperature (int/float): Ayarlanan sıcaklık değeri.
    
    Returns:
        dict: Klima cihazına gönderilecek veriyi içeren sözlük.
    
    Notes:
        - "name" alanı yalnızca görsel amaçlıdır ve gerekirse kaldırılabilir.
        - "groupId" alanı isteğe bağlıdır, string veya int olabilir.
    """

    def __generate_data_for_aircon_func(
        self, code_name, switch, mode, fanspeed, temperature
    ):
        try:
            group = self.__group_index[self.__aircon_key]
            name = str(code_name).capitalize().title().replace("_", " ")

            data = {
                "code": code_name,  # String            (KLIMA_GIRIS)
                "name": name,  # String            (Klima Giris)
                # NOTE: "name" attribute'u Sadece gorunus amacli, istenirse kaldirilabilir
                "switch": switch,  # Bool              (True / False)
                "mode": mode,  #
                "fanspeed": fanspeed,  #
                "temperature": temperature,  #
                "groupId": group,
            }  # String / Int      ("B" / 1)
            # NOTE: Group ID istege baglidir, string yada int olabilir

            return data

        except Exception as err:
            print("Generate Data Func Error = {}".format(err))
            return None

    """
    Veriyi belirlenen bir URL'ye POST isteği olarak gönderen bir fonksiyon.
    
    Args:
        data_to_send (dict): Sunucuya gönderilecek JSON formatındaki veri.

    Returns:
        response (requests.Response or None): Eğer istek başarılı olursa sunucudan gelen 
        cevabı döner. Aksi takdirde None döner.
    """

    def __send_request(self, data_to_send):
        try:
            if data_to_send is None:
                return None

            send_url = "{0}{1}/".format(self.__service_url, self.__location)
            print("\n\nLast Url = {}\n\n".format(send_url))

            # JSON formatındaki veriyi hazırla
            json_data = json.dumps(data_to_send)

            response = requests.post(
                send_url, data=json_data, headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            print(
                "Veri başarıyla gönderildi. HTTP Status Code = {}".format(
                    response.status_code
                )
            )
            print("Sunucu tarafından dönen veri = {}".format(response.json()))

            return response

        except requests.exceptions.RequestException as err:
            print("Request Exception = {}".format(err))
            return None

        except Exception as err:
            print("Send Request Main Error = {}".format(err))
            return None

    """
    Verilen URL ve port bilgisini kullanarak tam bir servis URL'si oluşturur.

    Args:
        url (str): Servisin ana URL adresi.
        port (int): Servisin dinlediği port numarası.

    Returns:
        str: Tam URL formatında servis adresi.
    """

    def __generate_url(self, url, port):
        return "http://{0}:{1}/".format(url, port)


def light_test(lounge_class):
    try:
        group_index = "d"
        place_index = 0
        switch = False
        dimming = 0

        # False degerler dimming degerleri 0 olarak gönderilecek

        # TODO: Tüm hepsini kapatabileceğimiz bir tag eklememiz gerekiyor
        # B - 1 Dikkatli bakilacak 2 ve 3 le calismiyor

        # TODO: Müziğin ekolayzırına göre dinamik veri yolla

        # Lede analog veri gönderebilir miyiz
        # Açma kapamada flicklenme (gidip gelme mevzusunu sor)
        # Amfi basamak için ekolayzırı sor
        # c2

        lounge_class.send_data_for_light_func(group_index, place_index, switch, dimming)

    except Exception as err:
        print("light_test Error = {}!".format(err))


def turn_off_all_lights(lounge_class):
    """
    Turns off all lights (in all lounge areas).
    Each device's 'switch' is set to False and 'dimming' to 0.
    """
    try:
        for group_index, devices in lounge_class._AmrLoungeClass__lounge_place.items():
            for place_index, device in enumerate(devices):
                code = device["code"]
                print(f"🔴 Turning OFF → Group: {group_index}, Device: {code}")
                lounge_class.send_data_for_light_func(
                    group_index=group_index,
                    place_index=place_index,
                    switch=False,
                    dimming=0,
                )
        print("✅ All lights have been turned OFF successfully.")

    except Exception as err:
        print(f"turn_off_all_lights Error = {err}!")


def turn_on_all_lights(lounge_class, default_dimming=120):
    """
    Turns ON all lights (in all lounge areas).
    If a device has a previous dimming value, it is used; otherwise, default_dimming is applied.
    """
    try:
        for group_index, devices in lounge_class._AmrLoungeClass__lounge_place.items():
            for place_index, device in enumerate(devices):
                code = device["code"]
                dimming_value = (
                    device["dimming"]
                    if device["dimming"] not in (None, 0)
                    else default_dimming
                )
                print(
                    f"🟢 Turning ON → Group: {group_index}, Device: {code}, Dimming: {dimming_value}"
                )
                lounge_class.send_data_for_light_func(
                    group_index=group_index,
                    place_index=place_index,
                    switch=True,
                    dimming=dimming_value,
                )
        print("✅ All lights have been turned ON successfully.")

    except Exception as err:
        print(f"turn_on_all_lights Error = {err}!")


def aircon_test(lounge_class):
    aircon_index = 1
    switch = True
    mode = 0
    fanspeed = 3
    temperature = 21

    lounge_class.send_data_for_aircon_func(
        aircon_index, switch, mode, fanspeed, temperature
    )


if __name__ == "__main__":

    try:
        service_url = (
            "10.10.10.244"  # "192.168.1.114"   #"10.10.10.46"    #"192.168.1.114"
        )
        port = 3001

        lounge_class = AmrLoungeClass(service_url, port)

        # light_test(lounge_class)
        # turn_on_all_lights(lounge_class)
        # aircon_test(lounge_class)

    except Exception as err:
        print("__main__ Error = {}!".format(err))
