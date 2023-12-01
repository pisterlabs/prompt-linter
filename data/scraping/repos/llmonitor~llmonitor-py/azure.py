import os
import requests
import json
import openai

from dotenv import load_dotenv
load_dotenv()

from llmonitor import monitor

openai.api_key = os.environ.get("AZURE_OPENAI_KEY")
openai.api_base = os.environ.get("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

monitor(openai)

# print(os.environ.get("AZURE_OPENAI_KEY"))
# print(os.environ.get("AZURE_OPENAI_ENDPOINT"))

deployment_name='Sonder16kTurbo' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
# model='gpt-3.5-turbo' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

# Send a completion call to generate an answer
print('Sending a test completion job')

completion = openai.ChatCompletion.create(engine=deployment_name, messages=[{"role": "user", "content": "Hello world"}])

print(completion.choices[0].message.content)