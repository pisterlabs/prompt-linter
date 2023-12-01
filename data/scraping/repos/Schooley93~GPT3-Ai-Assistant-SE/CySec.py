import openai 
import pyttsx3 
import SpeechRecognition as sr 
import time 


openai.api_key = " " 


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
  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=4000,
    n=1,
    stop=None,
    tempurature=0.5,
  )
  return response["choices"][0]["text"]

def speak_text(text):
  engine.say(text)
  engine.runAndWait()

def main():
  while True:
    
    print("Say 'Genius' to start recording your question...")
    with sr.Microphone() as source:
      recognizer = sr.Recognizer()
      audio = recognizer.listen(source)
      try:
        transcription = recognizer.recognize_google(audio)
        if transcription.lower() == "genius":

          filename = "input.wav"
          print("Say your question...")
          with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            source.pause_threshold = 1
            audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
            with open(filename, "wb") as f:
              f.write(audio.get_wav_data())
              
          
          text = transcribe_audio_to_text(filename)
          if text:
            print(f"You said: {text}")


            response = generate_response(text)
            print(f"GPT-3 says: {response}")

            tts = gs(text=response, lang='en')
            tts.save("sample.mp3")
            


            speak_text(response)
      except Exception as e:
        print("An error occurred: {}".format(e))

if __name__ == "__main__":
  main()              
             
