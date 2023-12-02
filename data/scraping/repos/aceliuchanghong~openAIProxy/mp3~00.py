import os
from openai import OpenAI
import httpx

proxyHost = "127.0.0.1"
proxyPort = 10809
proxies = {
    "http": f"http://{proxyHost}:{proxyPort}",
    "https": f"http://{proxyHost}:{proxyPort}"
}

client = OpenAI(http_client=httpx.Client(proxies=f"http://{proxyHost}:{proxyPort}"))
client.api_key = os.getenv("OPENAI_API_KEY")
audio_file = open("WeChat_20231007161725.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)
