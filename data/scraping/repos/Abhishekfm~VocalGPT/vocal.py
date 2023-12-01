import os
import time
import pyaudio
import speech_recognition as sr
import playsound
from gtts import gTTS
import openai


api_key = "Change this"

messages = [
    {"role": "system", "content": "You are a helpful AI Tutor."},
    {"role": "user", "content": "I am Abhishek, my date of birth is 27 jan 2000"},
    {"role": "assistant", "content": "Nice to meet you, Abhishek! Is there anything you would like me to help you with?"}
]

def update_chat(messages, role, content):
  messages.append({"role": role, "content": content})
  return messages



lang ='en'

openai.api_key = "your key"


guy = ""

while True:
    def get_adio():
        r = sr.Recognizer()
        with sr.Microphone(device_index=1) as source:
            audio = r.listen(source)
            said = ""

            try:
                said = r.recognize_google(audio)
                print(said)
                global guy
                guy = said

                if "Honey" in said:
                    words = said.split()
                    new_string = ' '.join(words[1:])
                    # messages = update_chat(messages, "user", new_string)
                    global messages
                    messages = update_chat(messages, "user", new_string)
                    print(new_string)
                    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
                    # completion = openai.Completion.create(model="text-davinci-003", max_token=100, prompt=said, temperature=0)
                    print(completion)
                    text = completion.choices[0].message.content
                    speech = gTTS(text = text, lang=lang, slow=False, tld="co.uk")
                    os.remove('welcome1.mp3')
                    speech.save("welcome1.mp3")
                    time.sleep(2)
                    playsound.playsound("welcome1.mp3")
                    messages = update_chat(messages, "assistant", text)
                    print(messages)
                else:
                    text = "Please Speak Friday before saying this " + said;
                    speech = gTTS(text=text, lang=lang, slow=False, tld="co.uk")
                    os.remove('welcome1.mp3')
                    speech.save("welcome1.mp3")
                    time.sleep(2)
                    playsound.playsound("welcome1.mp3")

            except Exception as e:
                print(e)


        return said

    if "stop" in guy:
        break


    get_adio()
