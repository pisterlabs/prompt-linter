# Paul Dou
# 10/05/23

# GPT file, handles OpenAI API functionality

# import python libraries
import openai # import OpenAI library
import re # import regex
import time

# regex
punct = re.compile("^[A-Z]$")

# initialize OpenAI key
openai.api_key = "" # insert OpenAI API key

# generate responses from OpenAI's GPT API by using input text as input
def remoteGPT(audioText):
    #GPTresponse = ""
    try: # if API key valid
        GPTout = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "user", "content": audioText}]) # create response by setting preferences
        GPTresponse = GPTout.choices[0].message.content # retrieve text response and save as string
        
        # vladVA info
        match re.sub(r'[^\w\s]','',audioText).lower().replace("whats","what is"):
            case "who are you":
                GPTresponse = "I'm VladVA, your personal voice assistant!"
            case "what is your name":
                GPTresponse = "I'm VladVA, your personal voice assistant!"
            case "what is your purpose":
                GPTresponse = "I'm VladVA, your personal voice assistant!"
            case "tell me about yourself":
                GPTresponse = "I'm VladVA, your personal voice assistant!"
            case "what do you do":
                GPTresponse = "I'm VladVA, your personal voice assistant!"
    except Exception as e: # API key
        GPTresponse = "Error: Current Plan Exceeded!"

    return GPTresponse # return GPT verbal diarrhea