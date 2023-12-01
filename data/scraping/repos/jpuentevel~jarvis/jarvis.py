import pyttsx3
import speech_recognition as sr
import openai

def main():
    r = sr.Recognizer()

    openai.api_key = "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"

    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-100)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume+0.50)
    engine.setProperty('voice', 'spanish')

    with sr.Microphone() as source:
        engine.say("Dime algo, por favor.")
        engine.runAndWait()
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            engine.say(f"Tú dijiste: {text}")
            engine.runAndWait()
        except:
            engine.say("No te entendí, lo siento.")
            engine.runAndWait()

if __name__ == "__main__":
    main()