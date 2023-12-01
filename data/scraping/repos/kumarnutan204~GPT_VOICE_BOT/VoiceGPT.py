import openai
import pyttsx3
import speech_recognition as sr 
import os
from dotenv import load_dotenv

#initialising text to speech engine
def configure():
    load_dotenv()
    
 

configure()

openai.api_key=os.getenv('API_KEY')

engine= pyttsx3.init()

def transcribe_audio_to_text(filename):
    recogniser= sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio= recogniser.record(source)
    try:
        return recogniser.recognize_google(audio)
    except:
        print("Skipping Unknown Error occured")
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]

def speak_text(text):
    engine.say(text)
    engine.runAndWait()
    
    
def main():
    while True:
        #wait for the user to say "genius" 
        print("Say 'Hello' to start recording your question")
        with sr.Microphone() as source:
            recognizer= sr.Recognizer()
            audio= recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "hello":
                    # record Audio
                    filename="input.wav"
                    print("Say your question....")
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source,phrase_time_limit=None, timeout=None)
                        with open(filename,"wb") as f:
                            f.write(audio.get_wav_data())
                            
                            
                        #transcribe audio to text
                        text = transcribe_audio_to_text(filename)
                        if text:
                            print(f"You said: {text}")
                            #Generate response using GPT-3
                            response= generate_response(text)
                            print(f"GPT-3 says: {response} ")
                            #read the response using text to speech
                            speak_text(response)
            except Exception as e:
                print("An error occured {}".format(e))
                
                
if __name__== "__main__":
    main()
                            