import cohere 
from cohere.classify import Example
import os
from dotenv import load_dotenv
load_dotenv()
cohere_api_key = os.environ['cohere_api_key']
print(cohere_api_key)
co = cohere.Client(cohere_api_key)
chat_conv = "Patient: Hi, how are you?"
response = co.generate(
        prompt=''.join(chat_conv)+"Doctor:",
        model='xlarge', max_tokens=20,   temperature=1.2,   k=0,   p=0.75,
        frequency_penalty=0,   presence_penalty=0, return_likelihoods='NONE',
        stop_sequences=["Patient:", "\n"]
    ).generations[0].text.strip()
print(response)