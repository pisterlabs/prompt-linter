import openai
import pyttsx3
import speech_recognition as sr
from api_secrets import API_KEY

openai.api_key = API_KEY

engine = pyttsx3.init() #initialize text to speech

r = sr.Recognizer() #initialize speech recognition
mic = sr.Microphone(device_index=1)


conversation = ""
user_name = "Phoenix"
bot_name = "John Cena"

while True:
    with mic as source:
        print("\nlistening...") #microphone is listening to the sound
        r.adjust_for_ambient_noise(source, duration=0.2) #getrid of bg noise
        audio = r.listen(source)
    print("no longer listening.\n")

    try:
        user_input = r.recognize_google(audio)
    except:
        continue

    prompt = user_name + ": " + user_input + "\n" + bot_name+ ": "

    conversation += prompt  # allows for context

    # fetch response from open AI api
    response = openai.Completion.create(engine='text-davinci-003', prompt=conversation, max_tokens=100)
    response_str = response["choices"][0]["text"].replace("\n", "")
    response_str = response_str.split(user_name + ": ", 1)[0] # getrid of extra msgs

    conversation += response_str + "\n"
    print(response_str)

    engine.say(response_str) #run text to speech
    engine.runAndWait()
