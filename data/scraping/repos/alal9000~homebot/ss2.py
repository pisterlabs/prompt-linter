import openai
import pyttsx3
import speech_recognition as sr

# settings and keys
openai.api_key = "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"
model_engine = "text-davinci-002"

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

r = sr.Recognizer()


def recognize_speech():
    speech = ""
    keyword = "Nova"
    while True:
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
        try:
            speech = r.recognize_google(audio)
            if keyword in speech:
                break
        except sr.UnknownValueError:
            pass
    return speech


def chatgpt_response(prompt):
    # send the converted audio text to chatgpt
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        n=1,
        temperature=0.7,
    )
    return response


def main():
    # run the program
    prompt = recognize_speech()
    print(prompt)
    responses = chatgpt_response(prompt)
    message = responses.choices[0].text
    print(message)
    engine.say(message)
    engine.runAndWait()


while True:
    main()
