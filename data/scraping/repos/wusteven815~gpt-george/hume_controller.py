import json
from env import OPENAI_KEY
from Hume.emotion_analysis import get_emotion
from gpt import GPTHandler
from audio import audio
from env import usingHume

api_url = "https://api.openai.com/v1/chat/completions"

file = './input.wav'
def get_result(voice, gpt):
    task = get_emotion(voice, file)
    data = json.loads(task)
    print(data)
    if len(data["emotions"]) != 0 and usingHume:
        emotions_text = " and ".join(data["emotions"]).lower()
        prompt = voice + " Reply to me as if I spoke with " + emotions_text + " in my voice."
    else:
        prompt = voice
    response = gpt.request(prompt)
    return response
