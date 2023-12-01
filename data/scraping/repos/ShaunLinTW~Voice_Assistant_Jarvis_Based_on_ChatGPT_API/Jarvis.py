import openai
import pyttsx3
import speech_recognition as sr
import sys
import threading

# Set up the ChatGPT API client
openai.api_key = "YOUR_API_KEY"

# Set up the text-to-speech engine
engine = pyttsx3.init()

#get the available voices
voice = engine.getProperty('voices')

# Set the voice to use
engine.setProperty('voice', voice[0].id)

# Set the volume
engine.setProperty('volume', 1.0)

# Set the rate at which the words are spoken
engine.setProperty('rate', 150)

# Set up the speech recognition engine
r = sr.Recognizer()

def speak(text):
  engine.say(text)
  engine.runAndWait()

def listen():
  with sr.Microphone() as source:
    audio = r.listen(source)
  try:
    text = r.recognize_google(audio)
    return text
  except Exception as e:
    print("Error: " + str(e))
    return None

def generate_response(prompt):
  completions = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.5,
  )

  message = completions.choices[0].text
  return message

speak("Hello, I am Jarvis. How can I help you today?")

while True:
  prompt = listen()
  if prompt is not None:
    if prompt == "thank you for your help":
      # Exit the program
      sys.exit()
    response = generate_response(prompt)
    speak(response)

    # Set up a timer to interrupt the text-to-speech engine after 10 seconds
    timer = threading.Timer(10.0, engine.stop)
    timer.start()

    # Speak the response
    response = generate_response(prompt)
    speak(response)

    # Cancel the timer if the response finishes speaking before it expires
    timer.cancel()
  else:
    speak("I'm sorry, I didn't understand that.")
