from record import SpeechRecognizer
from speak import Speak
import os
import openai
from chatbot import Chatbot
from prompt import Prompt

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')

listen = SpeechRecognizer()

speak = Speak()

chatbot = Chatbot(api_key=OPEN_AI_KEY, temprature = "0.5",base_prompt="")

print("Say something!")
while True:
    speak.speak("开始录音")
    message = listen.listen()
    speak.speak("录音结束")
    print(f"Recognized: {message}")
    # Start chat
    resp = chatbot.ask(message)
    response = resp["choices"][0]["text"]
    print(f"Response: {response}")
    speak.speak(response)
