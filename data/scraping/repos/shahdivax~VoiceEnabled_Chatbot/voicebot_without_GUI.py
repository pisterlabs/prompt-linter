import openai
import speech_recognition as sr
import pyttsx3

# set up OpenAI API credentials
openai.api_key = "Your_API_Key"

# initialize text-to-speech engine
engine = pyttsx3.init()


# define function to capture user's voice input
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)
    try:
        text_input = r.recognize_google(audio)
        print("You said: ", text_input)
        return text_input
    except:
        print("Could not recognize input.")
        return ""


# define function to generate text response using OpenAI API
def generate_response(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=input_text,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


# define function to convert text response to speech
def generate_speech_response(text_response):
    engine.say(text_response)
    engine.runAndWait()


# main loop to continuously listen for user input and generate responses
while True:
    user_input = get_voice_input()
    if user_input:
        response = generate_response(user_input)
        generate_speech_response(response)
