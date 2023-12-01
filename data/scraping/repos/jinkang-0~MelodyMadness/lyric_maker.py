import os
import openai
import re
from dotenv import load_dotenv, dotenv_values
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Initialize Environment Variables
load_dotenv()
keys = dotenv_values(".env.local")

# Initialize Mongo Database Connection
uri = keys["MONGO_URI"]
client = MongoClient(uri, server_api=ServerApi("1"))
database = client["ai_musician"]
collection = database["musicians"]

# Initialize AI Variables
openai.api_key = keys["OPEN_AI_KEY"]

def scrape_songs(artist):
    lyrics = ""
    attributes = collection.find({"name":f"{artist}"})
    for attribute in attributes:
        for song, lyric in zip(attribute["songs"], attribute["lyrics"]):
            lyrics += f"EXAMPLE SONG: {song}\nEXAMPLE LYRICS: {lyric}"
            lyrics += "\n"

    return lyrics

def get_artists():
    array = []
    attributes = collection.find({})
    for attribute in attributes:
        array.append(attribute["name"])
    return array

def generate_lyric_prompt(artist):
    return f'''Pretend that you are an up and coming artist and create a totally unique hit song inspired by {artist}.\nHere are some of their greatest works for inspiration!
    {scrape_songs(artist)}
        '''

def generate_melody_prompt():
    music_prompt = "Generate a melody in the form of a list of (pitch, duration) pairs in Python Syntax, where the pitch uses MIDI standards and the duration represents the number of quarter notes. Use a pitch of 0 to specify rest"
    requirements = " REQUIREMENTS: Make sure that the melody stays between MIDI pitch 50 and MIDI pitch 100, that the melody is at least 20 notes in length, and satisfies: "

    return music_prompt + requirements

def chat_response(prompt):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt= prompt,
      temperature=1,
      max_tokens=800,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response.choices[0].text

def generate_melody(prompt):
    initial_response = chat_response(generate_melody_prompt() + prompt)
    regex = re.search("(\[[\s\S]+\])", initial_response)
    harmonic_array = eval(regex.group())
    return harmonic_array

def generate_lyrics(artist):
    return chat_response(generate_lyric_prompt(artist))
