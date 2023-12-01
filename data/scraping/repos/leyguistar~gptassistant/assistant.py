#!/usr/bin/env python3
import openai
import json
import sys
import subprocess
import openai.error
"""
Project: gptassistant
Description: AI assistant with openai api

Author: David Sanchez
Contact: davida.sanchez@ciens.ucv.ve

Usage: assistant.py [prompt]


"""


if(len(sys.argv) < 2):
    print('Usage: assistant prompt')
    sys.exit()


with open('config.json','r') as h:
    config = json.load(h)
with open('prompts/assistant.txt','r') as h:
    prompt = h.read()
with open('prompts/help.txt','r') as h:
    help = h.read()
 

openai.api_key = config['openai_key']
user = sys.argv[1]
messages = [
        {"role": "system", "content": prompt},
        {"role": "assistant", "content": f'say "Hello, how can i assist you {config["name"]}?"'},
        {"role": "user", "content": 'What time it is?'},
        {"role": "assistant", "content": 'help'},
        {"role": "system", "content": help},
        {"role": "assistant", "content": 'date'},
        {"role": "system", "content": 'Fri 25 Aug 20:26:32 -04 2023'},
        {"role": "assistant", "content": 'say "its 8:26 pm"'},
        {"role": "user", "content": 'Turn off the lights and turn on the air conditioner'},
        {"role": "assistant", "content": 'loff; airon;"'},
        {"role": "user", "content": 'Play some music'},
        {"role": "assistant", "content": 'say "sure, what do you want to listen?"'},
        {"role": "user", "content": 'bad apple'},
        {"role": "assistant", "content": 'say "playing bad apple on youtube"; play "renai circulation"'},
        {"role": "user", "content": 'Stop the music'},
        {"role": "assistant", "content": 'stop'},
        {"role": "system", "content": 'command not found'},
        {"role": "assistant", "content": 'help'},
        {"role": "system", "content": help},
        {"role": "assistant", "content": 'say "sorry I don\'t know how to do that yet"'},
        {"role": "user", "content": 'que pelicula gano el oscar a mejor pelicula en 2020'},
        {"role": "assistant", "content": 'sgpt "pelicula ganadora del oscar a mejor pelicula en 2020"'},
        {"role": "system", "content": 'La película ganadora del Oscar a Mejor Película en 2020 fue "Parasite" (Parásitos) dirigida por Bong Joon-ho.'},
        {"role": "assistant", "content": 'say "La respuesta es parasite"'},
    ]
if(len(sys.argv) > 1):
  user = sys.argv[1]
  messages.append({"role": "user", "content": user})

#This loop will run max 3 times
max_allowed_tries = config['max_allowed_tries'] 
while max_allowed_tries > 0:
  max_allowed_tries -= 1
  try:
    chat_completion = openai.ChatCompletion.create(
      model=config['model'],
      messages=messages)
  except openai.error.ServiceUnavailableError as e:
    print(e)
    exit(-1)
  commands = chat_completion.choices[0].message.content
  print('commands gpt decided to use: ',commands)
  #reply help command:
  if(commands.startswith('help')):
      messages.append({"role": "system", "content": help})     
  else:
    if(commands.startswith('say')):
      sayCommand = True
      commands = './saylabs.py ' + commands[4:]

    else:
      sayCommand = False
    args = ['bash','-c',commands]
    process = subprocess.Popen(args, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    print(error)
    code = process.wait()
    if(code != 0):
      messages.append({"role": "system", "content": 'command not found'})
      continue
    #commands with no output and say commands will stop the loop
    if(output and not sayCommand):
      messages.append({"role": "system", "content": output.decode()})
    else:
      break