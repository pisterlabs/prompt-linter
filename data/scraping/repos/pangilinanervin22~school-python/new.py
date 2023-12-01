import openai
import os
import speech_recognition as sr
from gtts import gTTS
import flet as fl

openai.api_key = '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'


def ggs(visual: fl.Page):
    visual.theme_mode = fl.ThemeMode.DARK
    visual.window_height = 600
    visual.window_width = 700
    visual.title = 'My AI'
    text = fl.Text(value="This AI is helpful for blind people 😁,
                   color="orange", size=50)
    visual.controls.append(text)
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages)
    assistant_response = chat.choices[0].message.content
    text1 = fl.Text(f"BrainyAI🧠: {assistant_response}")
    visual.controls.append(text1)
    speak_response(assistant_response)
    messages.append({"role": "assistant", "content": assistant_response})
    text2 = fl.Text(f"{sentence['source']}")
    visual.controls.append(text2)


# fields
enter = fl.TextField(label="Enter your answer here")


def speak_response(response):
    tts = gTTS(text=response, lang='en')
    tts.save("response.mp3")
    os.system("response.mp3")


messages = [{"role": "system", "content": " I'm your assistant."}]


def ask_question(question):
    messages.append({"role": "user", "content": question})
    visual.update()


if __name__ == "__main__":
    fl.app(target=ggs)
    print("This AI is helpful for blind people 😁")
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        with microphone as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=0, phrase_time_limit=0)

        try:
            user_input = recognizer.recognize_google(audio)
            print(f"User👨‍🎓: {user_input}")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            ask_question(user_input)
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please try again.")
        except sr.RequestError as e:
            print(f"Sorry, I encountered an error: {e}")
        visual.update()
