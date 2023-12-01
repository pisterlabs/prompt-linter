import openai
import pyttsx3
engine = pyttsx3.init()

import converse.speak

voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_enGB_SusanM"
openai.api_key = "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"
messages = [
    {"role": "system", "content": "You are a kind helpful assistant.", }
]


def use_ai(message):
    if message:
        if "Stop" in message or "stop" in message:
            print("Jarvis: Glad to help you!")
            engine.say("Glad to help you!")
            engine.runAndWait()
            engine.stop()
        else:
            messages.append({"role": "user", "content": message})
            chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            reply = chat.choices[0].message.content
            print(f"Jarvis: {reply}")
            engine.setProperty('rate', 150)
            engine.setProperty('voice', engine.getProperty('voices')[2].id)
            engine.say(reply)
            engine.runAndWait()
            messages.append({"role": "assistant", "content": reply})
            engine.stop()
    else:
        print("Unable to recognize!")
        engine.stop()

