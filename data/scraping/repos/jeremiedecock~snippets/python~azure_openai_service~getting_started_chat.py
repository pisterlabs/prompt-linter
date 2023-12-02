#!/usr/bin/env python3

# See: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?pivots=programming-language-python&tabs=command-line

import openai

from credentials import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT     # Your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15'           # This may change in the future

# Send a completion call to generate an answer

print('Sending a test completion job')

start_phrase = 'Write a tagline for an ice cream shop. '
response = openai.Completion.create(engine=AZURE_OPENAI_DEPLOYMENT_NAME, prompt=start_phrase, max_tokens=10)
text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()

print(start_phrase + text)