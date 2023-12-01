import openai
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('pitch', 0.8)
recognizer = sr.Recognizer()

openai.api_key = ""
prompt= "hello there, in obi wan voice"
def create_response(text):
  response = openai.Completion.create(
  model="text-davinci-003",
  prompt=("Answer like the rapper drake." + str(text)),
  #prompt= ("Answer in the style of nietzsche but be bitter." + str(text)),
  temperature=0.9,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
  )
  return response.choices[0].text
###trying to make a function that will continue to chat with the user

while True:
    # Set up the microphone and listen for the user's voice
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)

    # Convert the audio to text
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        if "Jarvis" in text:
            # Respond to the user saying "Jarvis"
            engine.say("Yes, what can I do for you sir?")
            engine.runAndWait()
            # Listen for the user's next instructions
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                response=(create_response(text))
                engine.say(response)
                print(response)
                engine.runAndWait()
                engine.stop()
                break
                # Do something with the instructions (e.g., perform a task, etc.)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print("Sorry, there was an error processing your request: " + str(e))

# engine.say(create_response(text))
# engine.runAndWait()
# engine.stop()

