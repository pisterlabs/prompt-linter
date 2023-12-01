#GPT-3 Rans AI Voice Assistant
import pyttsx3 #pip install pyttsx3 - python text-to-speech
#GPT-3 powered AGI Chat Application: RANSFORD OPPONG: Aug 4,2023
import os
import openai #pip install openai
import gradio as gr #pip install gradio
import speech_recognition as sr #pip install SpeechRecognition == voice to text
import pyaudio
##sudo apt install espeak
openai.api_key = "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"
#command to tell the model how to arrange the inputs and outputs.
start_sequence = "\nAI:"
restart_sequence = "\Human: "
#initial input
prompt ="The following is a conversation with an AI Assistant. The Assistant is helpful, creative, clever and very friendly. \n\nHuman: Hello, who are you\nAI: I am an AI created by OpenAI. How may I assist you today?\nHuman: ",
#Speak Function: text to voice
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

speak("Hello , I'm Rans AI Voice Assistant, How can I help you?")
#voice to text
def STT():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio,language = "en-IN")
        print("Human said :" +query)
    except Exception as e:
        print(e)
        speak("Say that again please...")
        return "None"
    return query

def gpt_output(prompt):
    response = openai.Completion.create(
    model ="text-davinci-003",
    prompt = prompt,
    temperature = 0.9,
    max_tokens = 150,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0.6,
    stop = ["Human: ","AI: "]
    )
    data = response.choice[0].text
    # return data
    print(data)
    speak(data)
# a loop to take input from a user when the function is true
while True:
    query = STT()
    gpt_output(query)


##Solving pyaudio problem: 
###### sudo apt-get install portaudio19.dev
###### pip install pyaudio
