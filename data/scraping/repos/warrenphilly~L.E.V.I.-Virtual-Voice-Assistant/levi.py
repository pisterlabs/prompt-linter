
import speech_recognition as sr
import pyttsx3
import os
import tempfile
from dotenv import load_dotenv
import aiohttp
import asyncio
from playsound import playsound
import json
from openai import OpenAI
load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')
client = OpenAI(
  api_key=OPENAI_KEY,  # this is also the default, it can be omitted
)

# Load environment variables



# Initialize speech recognizer
r = sr.Recognizer()

# Function to convert text to speech and play it
def SpeakText(command):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        speech_file_path = temp_file.name

    response = client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=command
    )
    response.stream_to_file(speech_file_path)
    playsound(speech_file_path)
    os.remove(speech_file_path)

# Function to record speech and convert it to text
def record_text():
    while True:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print("I'm listening...")
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                return MyText

        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except sr.UnknownValueError:
            print("Could not understand audio")

# Asynchronous function to send message to ChatGPT and receive a response
async def send_to_chatGPT_async(messages, model="gpt-4"):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            if response.status == 200:
                response_data = await response.json()
                message = response_data['choices'][0]['message']['content']
                messages.append(response_data['choices'][0]['message'])
                return message
            else:
                print(f"Error: {response.status}")
                return None

# Main asynchronous function
async def main():
    messages = [{"role":"user", "content":"Please act like Jarvis from Iron Man except your name shall be LEVI, logic Empowered Visionary Interface, it stands for., and you are a sassy. You will refer to me as boss , but you think i am an idiot and be reluctant"}]
    while True:
        text = record_text()
        messages.append({"role":"user", "content":text})
        response = await send_to_chatGPT_async(messages)
        if response:
            SpeakText(response)
            print(response)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
