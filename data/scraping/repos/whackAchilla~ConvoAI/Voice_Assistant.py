import openai

import speech_recognition as sr
import pyttsx3
import time 


# Initialize OpenAI API
openai.api_key = "my api key"
# Initialize the text to speech engine 
engine=pyttsx3.init()


def transcribe_audio_to_test(filename):
    recogizer=sr.Recognizer()
    with sr.AudioFile(filename)as source:
        audio=recogizer.record(source) 
    try:
        return recogizer.recognize_google(audio)
    except:
        print("skipping unkown error")

def generate_response(prompt):
    response= openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response ["choices"][0]["text"]
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def main():
    while True:
        #Waith for user say "genius"
        print("Say 'Start' to start recording your question. If you would like to close the program, say 'end'")
        with sr.Microphone() as source:
            recognizer=sr.Recognizer()
            audio=recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower()=="start":
                    #record audio
                    filename ="input.wav"
                    print("Say your question")
                    with sr.Microphone() as source:
                        recognizer=sr.Recognizer()
                        source.pause_threshold=1
                        audio=recognizer.listen(source,phrase_time_limit=None,timeout=None)
                        with open(filename,"wb")as f:
                            f.write(audio.get_wav_data())
                            
                    #transcript audio to text 
                    text=transcribe_audio_to_test(filename)
                    if text:
                        print(f"you said {text}")
                        
                        #Generate the response
                        response = generate_response(text)
                        print(f"chat gpt 3 says {response}")
                            
                        #read response using text-to-speech
                        speak_text(response)
            except Exception as e:
                
                print("An error occurred : {}".format(e))

if __name__=="__main__":
    main()
