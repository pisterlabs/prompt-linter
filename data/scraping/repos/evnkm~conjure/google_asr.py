import speech_recognition as sr
import openai
import gtts
from playsound import playsound
from gtts import gTTS
from io import BytesIO

def Speak(text):
    # Generate the speech audio and store it in mp3_fp
    mp3_fp = BytesIO()
    #intro = "Greetings, this is a United States Army device, my purpose is to assess your injury and report back to my home base, and then proceed with further instructions."
    tts = gTTS(text, lang='en')
    tts.write_to_fp(mp3_fp)

    # Save the speech audio to a temporary file
    mp3_fp.seek(0)
    with open('temp.mp3', 'wb') as f:
        f.write(mp3_fp.read())

    # Play the audio using playsound
    playsound('temp.mp3')


def transcribe_speech():
    '''
    This function uses the microphone to listen for 10 seconds and then returns the transcribed text. 
    '''

    # Create a recognizer object
    recognizer = sr.Recognizer()

    # Set the duration for listening
    duration = 10  # Number of seconds to listen

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")
        # Adjust for ambient noise for better recognition
        recognizer.adjust_for_ambient_noise(source)

        # Record audio for the specified duration
        audio = recognizer.listen(source, timeout=duration)

    print("Finished recording.")

    try:
        # Recognize the speech using the default API
        transcript = recognizer.recognize_google(audio)
        return transcript
    except sr.UnknownValueError:
        return "Speech recognition could not understand audio"
    except sr.RequestError:
        return "Could not request results from the speech recognition service"


def ask_question(prompt):
    # Set up OpenAI API credentials
    openai.api_key = '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'

    # Set the GPT-3.5 model name
    model_name = "gpt-3.5-turbo"

    # Generate the question using GPT-3.5 chat model
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant in the year 2023."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )

    # Extract and return the generated answer
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def main():

    intro = ("Hi, hackMIT testing! Please say something hehehe")
    Speak(intro)

    # Get the transcribed text
    question = transcribe_speech()
    print(question)

    prompt = "Insert prompt here. This will be info about the image."

    # Generate the question and get the answer
    answer = ask_question(prompt)
    Speak(answer)



if __name__ == '__main__':
    main()
