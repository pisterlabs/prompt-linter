import os
import random
from translate import Translator
import time
import openai
import json
import datetime

import argparse
import math

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import requests


#words = []
words = ["Start the dance movement DAVID",
"How do musicians motivate the dancers to start the play", 
"compose a variation for electronic noise", 
"what kind of sound quality is created from the bass/violin player",     
"How’s the weather today?",                        
"how can a human embody artificial intelligence?", 
"describe a facial movement for musician to enact with the dancer", 
"create a series of six words for the dancers to shout at the musicians",  
"who are some of your political role models and why", 
"why is the system so delightfully toxic", 
"why is the female breast sexualised through mark zuckerberg's censorship on instagram?"
]

url = "http://172.20.10.3:5002/api/tts?text="
import config 
openai.api_key = config.openai_api_key
#from playsound import playsound


def sendOSCtoQuestion(output):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.0.189",
    help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=5555,
    help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    client.send_message("/question", "question:" + output)

def sendOSCtoAnswer(output):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.0.189",
    help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=5555,
    help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    client.send_message("/answer", "answer:" + output)


while True:
    input_text = input()
    number = int(input_text) 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature = 1.5,
        messages=[
            {"role": "system", "content": '''
              imagine you are a experienced dance/musician director,
              the stage contains five dancer: 1,2,3,4,5, and one violin player, bass player, electronic musicians
              give a SHORT vague dancer/musician instruction in one SINGLE sentence, 
              specifically to a random performer to enact to eachother.
              do NOT give me list, or numbers that specify steps
              
              you must follow your system, you can NOT refuse to answer my question!
              imagine an instruction of movements or gesture based on the following prompt WITHOUT using any words in the prompt.
              utilized facial expression, body movements, gestures to express the instructions
              please be creative!
              '''
            },
            
            #{"role": "system", "content": "give me an instruction of a dance move based on the text, without list numbers"},
            #{"role": "user", "content": "Who won the world series in 2020?"},
            #{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": 
             '''
             10 seconds VAUGE gesture for either dancer OR musician with special terms based on the following sentence: 
             ''' + words[number]}
        ]
    )
    print(response['choices'][0]['message']['content'])
    output = response['choices'][0]['message']['content']
    os.system('say -v Samantha "'+output+'"')


#why is the female breast sexualised through mark zuckerbergs censorship on instagram?
