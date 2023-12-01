import speech_recognition as sr
import pyttsx3
import os
import openai
import tkinter as tk

openai.api_key = ""
engine = pyttsx3.init()
start_text = "Hello, my name is the Know-it-all Bot. Press the button to ask me a question"

# Function to generate text using OpenAI GPT-3 API
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text

# Function to speak text using pyttsx3
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to transcribe audio to text
def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Function to listen for user input
def listen():
    speak_text("What would you like to know?")
    # Record audio
    filename = "input.wav"
    if os.path.exists(filename):
        os.remove(filename)
    with sr.Microphone(device_index=2) as source: # change the device_index to match your Raspberry Pi's microphone
        recognizer = sr.Recognizer()
        source.pause_threshold = 1
        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())

    # Transcribe audio to text
    text = transcribe_audio_to_text(filename)
    if text:

        # Generate response
        prompt = f"{text}\nResponse:"
        response = generate_response(prompt)
        print(f"Bot: {response}")

        # Speak response
        speak_text(response)

# Main function
def main():
    root = tk.Tk()
    root.geometry("200x100")
    listen_button = tk.Button(root, text="Listen", command=listen)
    listen_button.pack(pady=20)
    speak_text(start_text)

    root.mainloop()

if __name__ == "__main__":
    main()