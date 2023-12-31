import boto3
import subprocess
import requests
import json
import os
import threading
import telebot
import  sys
import re

with open('../.openapi_credentials') as f:
    contents = f.read()

'''
This codeThis code is iterating over each line in the `contents` string, which is split by the newline character (`\'\n\'`).
- generated by stenography autopilot [ 🚗👩‍✈️ ]
'''
for line in contents.split('\n'):
    if line.startswith('api_key='):
        API_KEY = line[len('api_key='):]
    elif line.startswith('bot_token='):
        BOT_TOKEN = line[len('bot_token='):]

# Open api autentication files in ~/.openapi_credentials
# api_key=
# api_secret=None

# Amazon Poly credentials in ~/.aws/credentials
# [default]
# aws_access_key_id = 
# aws_secret_access_key = 
# region=us-east-1

# Models: text-davinci-003,text-curie-001,text-babbage-001,text-ada-001
MODEL = 'gpt-3.5-turbo'

# Defining the bot's personality using adjectives
BOT_PERSONALITY = 'Resuma o texto para Português do Brasil: '

#Define response file
RESPONSE_FILE = './responses/responseGPT'
CHAT_ID= "-1001899083389"
QUEUE_FILE = 'queue.txt'
MP3_PLAYER = 'afplay -r 1.5'

# Define Prompt file
'''
This codeThis code is checking if there are commandline arguments provided when running the script. If no argument is provided, it reads a file called "queue.txt" and assigns the first line of that file to the variable `PROMPT_FILE`. It then reads all lines from the file into a list called `lines`, and writes all lines except for the first one back to the same file.
- generated by stenography autopilot [ 🚗👩‍✈️ ]
'''
if len(sys.argv) < 2:
    print("Não foi fornecido argumento, usando lista queue.txt")
    with open(QUEUE_FILE, 'r') as file:
            PROMPT_FILE = file.readline().strip()

    with open(QUEUE_FILE, 'r') as file:
        lines = file.readlines()

    with open(QUEUE_FILE, 'w') as file:
        file.writelines(lines[1:])
else:
    PROMPT_FILE = sys.argv[1]

'''

- generated by stenography autopilot [ 🚗👩‍✈️ ]
'''

import re

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # símbolos e pictogramas
                               u"\U0001F680-\U0001F6FF"  # transporte e símbolos de mapa
                               u"\U0001F1E0-\U0001F1FF"  # bandeiras de países
                               u"\U00002702-\U000027B0"  # símbolos diversos
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def polly_speak(response_file):
    # Crie uma instância do cliente da API Polly
    polly_client = boto3.client('polly')

    # Defina as configurações de voz e linguagem
    voice_id = 'Camila'
    language_code = 'pt-BR'
    engine = 'neural'


    # Defina o texto que será sintetizado em fala
    with open(response_file + '.txt', "r") as file:
        text = file.read()

    # Use o método synthesize_speech() da API Polly para sintetizar o texto em fala
    response = polly_client.synthesize_speech(
        OutputFormat='mp3',
        Text=text,
        VoiceId=voice_id,
        LanguageCode=language_code,
        Engine=engine
        )

    # Salve o áudio sintetizado em um arquivo audio
    audio_file = response_file + ".mp3"
    with open(audio_file, 'wb') as f:
        f.write(response['AudioStream'].read())
        f.close()
    audio_send(CHAT_ID, audio_file)

    command = MP3_PLAYER + " " + audio_file
    #subprocess.run(command, shell=True)

# 2a. Function that gets the response from OpenAI's chatbot
'''

- generated by stenography autopilot [ 🚗👩‍✈️ ]
'''
def open_ai(prompt):
    # Make the request to the OpenAI API
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {API_KEY}'},
        json={'model': MODEL, 'messages': prompt, 'temperature': 0.01}
    )

    result = response.json()
    final_result = ''.join(choice['message'].get('content') for choice in result['choices'])
    return final_result

'''
This codeThis code defines a function called `audio_send` that sends an audio file to a Telegram bot chat.
- generated by stenography autopilot [ 🚗👩‍✈️ ]
'''
def audio_send(chat_id, output_audio):
    """
    Sends an audio file to a Telegram bot chat. 

    :param OUTPUT_AUDIO: a string representing the path to the audio file
    :param chat_id: an integer representing the chat id
    :return: None
    """
    bot = telebot.TeleBot(BOT_TOKEN)
    audio_file=open(output_audio,'rb')
    bot.send_audio(chat_id, audio_file)

'''

- generated by stenography autopilot [ 🚗👩‍✈️ ]
'''
def telegram_bot_sendtext(bot_message,chat_id):
    data = {
        'chat_id': chat_id,
        'text': bot_message
    }
    response = requests.post(
        'https://api.telegram.org/bot' + BOT_TOKEN + '/sendMessage',
        json=data
    )
    return response.json()

# Run the main function
'''

- generated by stenography autopilot [ 🚗👩‍✈️ ]
'''
if __name__ == "__main__":
    with open(PROMPT_FILE, "r") as file:
        prompts = remove_emojis(file.read().strip())
        contador_linhas = len(prompts.split('\n'))

    print(contador_linhas)
    if contador_linhas > 1:

        promptList = prompts.split('\n\n') 

        for index, prompt in enumerate(promptList):
            string_formatada = "{:03d}".format(numero)
            if len(prompt) > 10:
                bot_response = open_ai([{'role': 'user', 'content': f'{BOT_PERSONALITY} {prompt}'}])
                
                bot_response = bot_response.replace('\n', '. ').strip()
                bot_response = bot_response.replace('..', '.')

                with open(RESPONSE_FILE + str(string_formatada) + ".txt", "w") as file:
                    file.write(bot_response)
                
                polly_speak(RESPONSE_FILE + str(string_formatada))
                os.remove(RESPONSE_FILE + str(string_formatada) + ".txt")
                os.remove(RESPONSE_FILE + str(string_formatada) + ".mp3")
            bot_response = ""
    else:
        telegram_bot_sendtext(prompts,CHAT_ID)
    os.remove(PROMPT_FILE)  

