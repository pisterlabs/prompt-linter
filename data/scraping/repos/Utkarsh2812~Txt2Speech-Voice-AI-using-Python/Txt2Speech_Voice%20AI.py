import openai
import pyttsx3
import speech_recognition as sr

openai.api_key = 'Your API Key'
engine = pyttsx3.init()


def transcribe_audio_to_txt(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print('Skipping Unknown Error')


def generate_response(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response['choices'][0]['text']


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        # Start with Book
        print('Say "Book" to begin recording...')
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300  # Adjust input sensitivity as needed
            audio = recognizer.listen(source, phrase_time_limit=None,
                                      timeout=None)  # Set phrase_time_limit to None for unlimited recording time
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == 'book':
                    # Record Audio
                    filename = 'Input.wav'
                    print('What is your Query?..')
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        recognizer.energy_threshold = 300  # Adjust input sensitivity as needed
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None,
                                                  timeout=None)  # Set phrase_time_limit to None for unlimited
                        # recording time
                        with open(filename, 'wb') as f:
                            f.write(audio.get_wav_data())

                    # Transcription of audio to txt
                    text = transcribe_audio_to_txt(filename)
                    if text:
                        print(f"Word Spoken: {text}")

                        # Generation of Output using GPT-3
                        response = generate_response(text)
                        print(f"Word Recognized: {response}")

                        # Reading Response Using Txt2Speech
                        speak_text(response)
            except Exception as e:
                print('Error occurred: {}'.format(e))


if __name__ == '__main__':
    main()
