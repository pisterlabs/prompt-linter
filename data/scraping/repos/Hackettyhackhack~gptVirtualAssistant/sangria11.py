import openai
import pyttsx3
import speech_recognition as sr
import time
import datetime

openai.api_key = ""  # Update with valid OpenAI API key

engine = pyttsx3.init()

def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except: 
        print('Skipping unknown error')

def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003", 
            prompt=prompt,
            max_tokens=4000,
            n=1, 
            stop=None,
            temperature=0.6,
        )
        return response["choices"][0]["text"]
    except Exception as e:
        print("Failed to generate response: {}".format(e))

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def get_current_time():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M")
    return current_time

def main():
    while True:
        print("Say 'Genius' to start your question or 'Stop' to terminate...")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "genius":
                    # Record audio
                    filename = "input.wav"
                    print("Ask your question")
                    with sr.Microphone() as source: 
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f: 
                            f.write(audio.get_wav_data())

                    # Transcribe audio to text
                    text = transcribe_audio_to_text(filename)
                    if text: 
                        print(f"You: {text}")

                        # Check if asking for current time
                        if "what time is it" in text.lower():
                            current_time = get_current_time()
                            print(f"The current time is: {current_time}")
                            speak_text(f"The current time is: {current_time}")
                        else:
                            # Generate response using GPT-3
                            response = generate_response(text)
                            if response:
                                print(f"Mr GPT: {response}")

                                # Read response using TTS
                                speak_text(response)
                elif transcription.lower() == "stop":
                    print("Terminating the program...")
                    break
            except Exception as e:
                print("An error occurred: {}".format(e))

if __name__ == "__main__":
    main()
