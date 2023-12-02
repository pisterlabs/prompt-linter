#Voice Assistant Echo powered by pyttsx3 (text to speech engine) and speech recognition engine
import pyttsx3
import speech_recognition as sr
import openai
import json
import os
from datetime import datetime

#Get Time
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
current_date = now.strftime("%Y-%m-%d")

print("Current Time =", current_time)
print("Current Date =", current_date)


# Initialize text-to-speech engine
engine = pyttsx3.init()

# List available voices
voices = engine.getProperty('voices')

# Set default VOICE!!!
siri_voice = "com.apple.speech.synthesis.voice.samantha"
#siri_voice= "com.apple.speech.synthesis.voice.tom"  #male

# Set rate and volume
#engine.setProperty('rate', 150)  # Experiment with this
engine.setProperty('volume', 1.0)  # Max volume
engine.setProperty('voice', siri_voice)

#ECHO SAYS
def speak(text, voice_id):
    engine.setProperty('voice', voice_id)
    engine.say(text)
    engine.runAndWait()
    
# Initialize speech recognition
recognizer = sr.Recognizer()

#API KEY: protect it
KEY = ''

# Initialize OpenAI API
# openai.api_key = os.getenv(KEY)  # Replace with your own API key if you prefer hardcoding (not recommended)
openai.api_key = KEY

#GET VOICE Command
def get_audio_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio_data = recognizer.listen(source)
        text = recognizer.recognize_google(audio_data)
        return text

#GPTransformer Generates -- Change the Model; Create custom one or import pretrained transformer
"""Other pretrained transformer from microsoft: Dialo...

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# chat for 5 lines example:
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

"""


def generate_response(text):
    # Generate response using OpenAI API
    prompt = f"{text}"
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can update this to "text-davinci-turbo" based on your API availability
        prompt=prompt,
        max_tokens=50
    )
    #might need to delete the Echo says
    response = "Echo says:" + response.choices[0].text.strip()
    
    return response


#LOG HISTORY
def append_to_file(role, text):
    with open("conversation_history.txt", "a") as f:
        f.write(f"{role}: {text}\n")

"""# Check Available Voices
voices = engine.getProperty('voices')
for index, voice in enumerate(voices):
    print(index, voice.id)"""


#RUN... response happens outside the main loop -> in the generate_response(txt) function
while True:
    try:
        # Get audio input
        user_input = get_audio_input()
        print(user_input)
        
        # Generate response
        response = generate_response(user_input)
        
        #Add the Date
        append_to_file("\n Hello World!", f"Date: {current_date} | Time: {current_time} \n ")
        
        # Append user input to file
        append_to_file("User", user_input)

        # Append assistant response to file
        append_to_file("Assistant", response)

        # Speak the response
        print(response)
        speak(response, siri_voice)

    except Exception as ex:
        print(f"An error occurred: {ex}")
