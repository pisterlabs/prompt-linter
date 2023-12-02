""" 
Created on : 02/05/23 4:27 pm
@author : ds  
"""
import openai
from config import Open_AI

openai.api_key = Open_AI.key

models = openai.Model.list()
print(models)

while models:
    for model in models['data']:
        print(model['id'])
    models = models.get('next')

completion = openai.Completion.create(model="ada", prompt="Hello world")
print(completion.choices[0].text)


audio_file = open("New Recording 3.mp3", 'rb')
transcript = openai.Audio.translate("whisper-1", audio_file)

print(transcript)
