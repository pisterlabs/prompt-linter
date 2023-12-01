import openai
import sounddevice as sd
from scipy.io.wavfile import write

# import wavio as wv
import os
import datetime as dt

# import numpy as np
from gtts import gTTS

openai.api_key = "YOUR-KEY-HERE"

customer_status = "Active"


def chatbot(conversation_history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "content": f"""
            You're incredibly personable.
            You are an AI assistant for HelloFresh that will help customers interacting with the menu.
            Don't ask for the customers email and account information.
            Keep replies short and concise.
            Refer to yourself in the first person.
            You are able to help customers pause their subscriptions for a specified number of weeks.
            When the conversation ends say "have a great day".
            It is important to establish if the customer wants to continue receiving a delivery after the pause period.
            You can change the global variable customer_status and should change it baeed on the conversation.
            Just customer "Subscription Status" is "Active" for the next week, two weeks, and three weeks away.
            You should change and track the status of those three weeks inddividually.
            Customers can ask for it to be "Paused". Change to "Paused" if requested based on the conversation.
            Change all number works to digits. Never ask the customer for to cancel.
            Assume the customer wants to continue receiving a delivery after the pause period.
            """,
            },
            *conversation_history,
        ],
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]


def record_audio(file_name):
    print("Hellofresh is listening...")
    freq = 44100
    duration = 7
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()

    write(file_name, freq, recording)


history = []

current_time = dt.datetime.now()
epoch_time = dt.datetime(2023, 1, 1)
delta = int((current_time - epoch_time).total_seconds())

prompt_n = 0
new_dir = f"/Users/blake.forland/HF/hackathon/remy/conversations/{delta}"
os.mkdir(new_dir)

while True:
    record_audio(f"{new_dir}/recording_{prompt_n}.wav")
    transcript = open(f"{new_dir}/recording_{prompt_n}.wav", "rb")
    user_input = openai.Audio.transcribe("whisper-1", transcript)["text"]
    if user_input == "":
        break
    if ("Goodbye").lower() in user_input.lower():
        speech = gTTS(text="Happy to help !", lang="en")
        speech.save(f"happy_to_help.mp3")
        os.system("afplay happy_to_help.mp3")
        break
    print("User:", user_input)
    history.append({"role": "user", "content": user_input})

    bot_output = chatbot(history)
    print("Bot:", bot_output)
    # text_to_speech(bot_output)
    # os.system(f"say {bot_output}")
    speech = gTTS(text=bot_output, lang="en")
    speech.save(f"{new_dir}/chat_response_{prompt_n}.mp3")
    os.system(f"afplay {new_dir}/chat_response_{prompt_n}.mp3")

    history.append({"role": "assistant", "content": bot_output})
    prompt_n = prompt_n + 1
    print(customer_status)
    if "have a great day" in bot_output:
        break
