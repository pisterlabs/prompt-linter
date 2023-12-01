import os
import requests
from chatgpt_wrapper import ChatGPT
import base64
import json

bot = ChatGPT()
response = bot.ask(input("> "))
print(response)

#If the code doesn't work although you did everything right - My apologise, you will need to
#insert api key from OpenAi, if you dont know how to do it - Ask chat gpt

token = "Input token of text-to-speech from google cloud "

url = "https://us-central1-texttospeech.googleapis.com/v1beta1/text:synthesize"
method = "POST"
headers = {
    "X-goog-api-key": f"{token}",
    "Content-Type": "application/json"
}


data = {
    "audioConfig": {
        "audioEncoding": "LINEAR16",
        "effectsProfileId": [
            "small-bluetooth-speaker-class-device"
        ],
        "pitch": 0,
        "speakingRate": 1
    },
    "input": {
        "text": response
    },
    "voice": {
        "languageCode": "en-US",
        "name": "en-US-Neural2-F"
    }
}





response = requests.post(url, headers=headers, json=data)
if response.status_code >= 300:
    print("error requesting to the google api")
    print(response.content)

content = json.loads(response.content)
audio = base64.b64decode(content["audioContent"])

with open("output.mp3", "wb") as f:
    f.write(audio)
    
print("wrote to file `output.mp3`")

# print(response.content)
