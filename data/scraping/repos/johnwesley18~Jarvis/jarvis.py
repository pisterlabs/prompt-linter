
import sounddevice as sd
import numpy as np
import speech_recognition as sr 
import pyttsx3
import speech_recognition as sr
import openai
import requests






def ask_gpt3(prompt):
    # Replace YOUR_OPENAI_API_KEY with your actual OpenAI API key
    openai.api_key = "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"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
    )

    if "choices" in response and len(response["choices"]) > 0:
        return response["choices"][0]["text"].strip()
    else:
        print("No valid response received from GPT-3.")
        return ""


def listen():
    mic_index = None
    audio_data = None

    # Choose the microphone index. You can customize this part if needed.
    mic_list = sd.query_devices()
    for index, device in enumerate(mic_list):
        if "microphone" in device["name"].lower():
            mic_index = index
            break

    if mic_index is None:
        print("No microphone found.")
        return ""

    print("Listening...")
    with sd.InputStream(samplerate=44100, channels=1, dtype=np.int16, device=mic_index) as stream:
        audio_data, _ = stream.read(44100 * 5)  # Adjust the number of samples to capture (5 seconds in this case)

    r = sr.Recognizer()
    
    try:
        audio_data = sr.AudioData(audio_data.tobytes(), 44100, 2)  # Convert numpy array to AudioData
        user_input = r.recognize_google(audio_data)  # Use Google Web Speech API for speech recognition
        print("You said:", user_input)
        return user_input.lower()
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
        return ""
    except sr.RequestError as e:
        print("Error occurred while requesting results from Google Web Speech API; {0}".format(e))
        return ""
    

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  # You can adjust the speech rate here
    engine.say(text)
    engine.runAndWait()


def main():
    print("Jarvis: Hi! How can I assist you today?")
    while True:
        user_input = listen()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Jarvis: Goodbye!")
            speak("Goodbye!")
            break

        response = ask_gpt3(user_input)
        print("Jarvis:", response)
        speak(response)


if __name__ == "__main__":
    main()
