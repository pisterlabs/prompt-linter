import os
from dotenv import load_dotenv
import openai


class Env:
    def __init__(self):
        load_dotenv()

    def openai_key(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment varables are missing.")
        openai.api_key = api_key
        return api_key

    def weather_key(self):
        weather_key = os.getenv("WEATHER_API_KEY")
        if not weather_key:
            raise ValueError("Environment variables are missing.")
        return weather_key

    def searper_key(self):
        searper_key = os.getenv("SERPER_API_KEY")
        if not searper_key:
            raise ValueError("Environment variables are missing.")
        return searper_key

    def wolframalpha_appid(self):
        wolframalph_appid = os.getenv("WOLFRAMALPHA_APP_ID")
        if not wolframalph_appid:
            raise ValueError("Environment variables are missing.")
        return wolframalph_appid

    def bing_map_key(self):
        bing_map_key = os.getenv("BING_MAP_KEY")
        if not bing_map_key:
            raise ValueError("Environment variables are missing.")
        return bing_map_key
