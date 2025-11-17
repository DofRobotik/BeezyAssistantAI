from typing import List, Dict


def get_tools_schema(station_names: List[str], device_codes: List[str]) -> List[Dict]:
    """
    Generates the JSON schema for Ollama tool calling.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Use this tool to search the internet for information you don't know (e.g., news, weather, exchange rates, specific facts). Do not use for internal mall queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query optimized for a search engine.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "navigate_to_station",
                "description": "Moves the robot to a specific station. IMPORTANT: ONLY CALL THIS AFTER THE USER HAS EXPLICITLY CONFIRMED THEY WANT TO GO THERE. Do not call this on the first request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "station_name": {
                            "type": "string",
                            "enum": station_names,
                            "description": "The exact station code (e.g., 'station_a').",
                        }
                    },
                    "required": ["station_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "control_iot_device",
                "description": "Controls smart devices (lights/aircon). Select the correct DEVICE_CODE based on the user's description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_code": {
                            "type": "string",
                            "enum": device_codes,
                            "description": "The exact device code (e.g., 'MUTFAK_GENEL').",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["turn_on", "turn_off"],
                            "description": "The action to perform.",
                        },
                    },
                    "required": ["device_code", "action"],
                },
            },
        },
    ]
