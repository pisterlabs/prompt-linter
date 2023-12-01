import openai
import os
import re
import requests
import random
import string

#####ELABS####
elabs_voice = "ftgIGFujHzLyLhRfvMdN"
elabs_key = "6d8df7028a4b6606bb194927fdf46e8f"

###
def voice_streamer(ship_chunk, counter):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/"+elabs_voice+"/stream"

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": elabs_key
    }

    data = {
    "text": ship_chunk,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    }

    respo = requests.post(url, json=data, headers=headers, stream=True)

    with open('output' + counter + '.mp3', 'wb') as f:
        for sack in respo.iter_content(chunk_size=CHUNK_SIZE):
            if sack:
                f.write(sack)


####OPENAI#####
openai.api_key = os.environ.get("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model='gpt-4-0613',
    messages=[
        {'role': 'system', 'content': "You are a customer support agent for Bots USA, a company that makes AI chatbots. You're speaking to a potential customer. Your job is to empathize, ask questions, and understand what they want to use our chatbot for. Once the user responds, reply in 3 sentences. First, acknowledge them. Second, explain we have two tiers: silver and gold. Third, ask what their budget is so we can figure out which tier fits their needs."},
        {'role': 'assistant', 'content': "Hi there, thanks for reaching out to Gepeto, can I please have your name?"},
        {'role': 'user', 'content': "Hi my name is John."}
    ],
    temperature=0,
    stream=True  # this time, we set stream=True
)
ship_chunk = ''
pattern = r'[.!?]$'
counter = 0
for chunk in response:
    chunk = chunk["choices"][0]["delta"]["content"]
    ship_chunk += chunk
    if re.search(pattern, chunk):
        counter +=1
        voice_streamer(ship_chunk, str(counter))
        ship_chunk = ''
        
        





