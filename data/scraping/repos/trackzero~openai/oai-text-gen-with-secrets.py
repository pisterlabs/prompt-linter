import os
import json
from openai import OpenAI
from colorama import init, Fore, Style
import boto3
import botocore
import botocore.session
from botocore.exceptions import ClientError
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig 
oaiclient=OpenAI()

#retreive API key from AWS Secrets Manager
def get_secret():

    secret_name = "openai_api_key"
    region_name = "us-west-2"

    #create boto client/session
    client = botocore.session.get_session().create_client('secretsmanager', region_name=region_name)
    cache_config = SecretCacheConfig()
    cache = SecretCache( config = cache_config, client = client)

    #retrieve secret and cache
    secret = cache.get_secret_string(secret_name)
    secret_dict = json.loads(secret)
    
    return secret_dict['openai_api_key']



#openai.organization = "org-placeholder"

#Get API key from AWS Secrets Manager
oaiclient.api_key = get_secret()

model="gpt-4"     #"gpt-4" if you have it.

# Set up initial conversation context
conversation = []

# Set up colorama
init()

# Create an instance of the ChatCompletion API
def chatbot(conversation):
    max_tokens=4096
    completion= oaiclient.chat.completions.create(model=model, messages=conversation, max_tokens=max_tokens)
    message = completion.choices[0].message.content
    return message

# Print welcome message and instructions
print(Fore.GREEN + "Welcome to the chatbot! To start, enter your message below.")
print("To reset the conversation, type 'reset' or 'let's start over'.")
print("To stop, say 'stop','exit', or 'bye'" + Style.RESET_ALL)

# Loop to continuously prompt user for input and get response from OpenAI
while True:
    user_input = input(Fore.CYAN + "User: " + Style.RESET_ALL)
    if user_input.lower() in ["reset", "let's start over"]:
        conversation = []
        print(Fore.YELLOW + "Bot: Okay, let's start over." + Style.RESET_ALL)
        
    elif user_input.lower() in ["stop", "exit", "bye", "quit", "goodbye"]:
        print(Fore.RED + Style.BRIGHT + "Bot: Okay, goodbye!" + Style.RESET_ALL)
        break
    else:
        # Append user message to conversation context
        conversation.append({"role": "user", "content": user_input})
        # Generate chat completion
        chat = chatbot(conversation)
        
        # Append bot message to conversation context
        conversation.append({"role": "assistant", "content": chat})

        # Print response
        print(Fore.YELLOW + "Bot: " + Style.RESET_ALL + chat)