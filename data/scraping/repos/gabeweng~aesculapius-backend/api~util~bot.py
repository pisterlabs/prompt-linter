from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib import parse
import json
from twilio.rest import Client
import cohere 
from cohere.classify import Example
import os
from dotenv import load_dotenv
load_dotenv()

try:
  cohere_api_key = os.environ['cohere_api_key']
  print(cohere_api_key)
  co = cohere.Client(cohere_api_key)
except:
  print("please enter the cohere_api_key")



examples =  [
  Example("I feel like no one loves me", "Self-harm"),  
  Example("I feel meaningless", "Self-harm"),  
  Example("I want to feel pain", "Self-harm"),  
  Example("I want everything to end", "Self-harm"),  
  Example("Why does no one love me?", "Self-harm"),  
  Example("My chest hurts really badly. Please help!", "Medical attention"),  
  Example("My arm is broken", "Medical attention"),
  Example("I have a giant cut on my leg!", "Medical attention"),    
  Example("I feel like I'm going to pass out", "Medical attention"),
  Example("I think I'm getting warts on my genitals. What does that mean", "Symptoms"),    
  Example("I have a slight fever and cough. What do I have", "Symptoms"),    
  Example("I have diarrea and muscle aches. What do you think I have", "Symptoms"),
  Example("I have a small headache and some trouble breathing. What does that mean", "Symptoms")
]

history = dict()
def bot(_inputs,user="Bob"):
    global history
    if user in history:
      print("NEW USER")
      chat_conv = history[user] + "Patient: " + _inputs + "\n"
    else:
      chat_conv = "Patient: " + _inputs + "\n"
    intent = co.classify(  
    model='medium',  
    inputs=[_inputs],  
    examples=examples)

    retintent = "None"
    print(intent.classifications[0].confidence)
    if (intent.classifications[0].confidence >= 0.95):
        retintent = intent.classifications[0].prediction
        print(retintent)
    if retintent == "Self-harm":
      SMSText("ALERT: A USER SAID: " + _inputs)
    response=co.generate(
    prompt=''.join(chat_conv)+"Family Doctor:",
        model='xlarge', max_tokens=20,   temperature=1.2,   k=0,   p=0.75,
        frequency_penalty=0,   presence_penalty=0, return_likelihoods='NONE',
        stop_sequences=["Patient:", "\n"]
    ).generations[0].text.strip().split("Patient:")[0]
    history[user] = "Patient: " + _inputs + "\n" +"Family Doctor: "+ response + "\n"
    return retintent, response

def SMSText(message):
  account_sid = os.environ['TWILIO_ACCOUNT_SID']
  auth_token = os.environ['TWILIO_AUTH_TOKEN']
  client = Client(account_sid, auth_token)

  message = client.messages \
                  .create(
                      body=message,
                      from_=os.environ['FROM_PHONE_NUM'],
                      to=os.environ['TO_PHONE_NUM']
                  )

  print(message.sid)