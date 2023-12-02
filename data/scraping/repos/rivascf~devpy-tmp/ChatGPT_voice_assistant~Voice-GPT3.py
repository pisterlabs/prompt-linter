import openai
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import re
# import time

# Set OpenAI API key
openai.api_key = "**-************************************************"
# Initialize text-to-speech engine


def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print('Skipping unknown error')


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]


def speak_text(text):
    # say = gTTS(text=text, lang='es', tld='com.mx')
    res = len(re.findall(r'\w+', text))
    print (f"In ChatGPT3 response are {len(text)} chars and {str(res)} words.")
    say = gTTS(text=text, lang='en')
    say.save('result_voice.mp3')
    playsound('result_voice.mp3')


def main():
    while True:
        # Wait for user to say "genius"
        print("Say 'Genius' to start recording your question...")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                print(type(transcription))
                if transcription.lower() == 'genius':
                    # Records audio
                    filename = "input.wav"
                    print("Say your question...")
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())

                        # Transcribe audio to text
                        text = transcribe_audio_to_text(filename)
                        if text:
                            print(f"You said: {text}")

                            # Generate response using Chat GPT-3
                            response = generate_response(text)
                            print(f"GPT-3 says: {response}")

                            # Read response using text-to-speech
                            speak_text(response)
            except Exception as e:
                print("An error has occurred: {}".format(e))


if __name__ == "__main__":
    main()
