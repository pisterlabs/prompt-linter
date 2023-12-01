import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
import openai

# read csv file

df = pd.read_csv('./data/youtube-desk-setup-raw-data.csv')
print(df.shape)

# do some data cleaning
df['transcript_text'] = df['transcript_text'].str.replace('\n', ' ')
df['transcript_text'] = df['transcript_text'].str.replace('\t', ' ')
df['transcript_text'] = df['transcript_text'].str.replace('[Music]', '')
df['transcript_text'] = df['transcript_text'].str.replace('[Applause]', '')

# remove rows with less than 100 words
df['word_count'] = df['transcript_text'].apply(lambda x: len(x.split()))
df = df[df['word_count'] > 500]

# remove rows published before 2023
df['release_date'] = df['release_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df = df[df['release_date'] >= datetime(2023, 1, 1)]

print(df.shape)
df.to_csv('./data/cleaned-youtube-desk-setup.csv', index=False)

# openai api
load_dotenv(find_dotenv())
openai.api_key = os.environ.get("OPENAI_API_KEY")

def openai_api(text:str) -> str:
    # openai api
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": """You will be provided with Desk setup youtube video transcripts, and your task is to extract what items are mentioned in video: 
                1. you must only extract items mentioned in the video 
                2. you must find 3 main items in the video, computer, mouse and keyboard, if its not found in video, say \"NA\" 
                3. if other desk items is mentioned also put them in the output, monitor, lights, desk, charger, computer dock etc.
                4. if same category have multiple items put them in a string with comma separated.
                5. your output format should be in json\n\n
                here is one example of respond:
                ```
                {"computer": "14 inch MacBook Pro", "mouse": "Logitech MX Master 3s", "keyboard": "Logitech MX mechanical mini", "monitor": "Apple Studio display, BenQ PD 3220u", "lights": "Elgato key light air", "desk": "Ikea countertop, Alex drawers, Carly kitchen countertop", "charger": "CalDigit TS4 Thunderbolt Hub", "computer dock": "Book Arc by 12 South", "neon sign": "custom neon sign by Illusion Neon", "acoustic panels": "gig Acoustics panels", "desk chair": "Autonomous Ergo chair plus", "scanner": "Fujitsu scanner", "charging stand": "Pataka charging stand", "pen": "Grovemade pen", "sticky notes": "sticky notes", "webcam": "Opal C1", "microphone": "Shure MV7", "audio interface": "Apollo twin X", "speakers": "Yamaha HS5", "headphones": "Rode NTH100s", "mic arm": "Rode PSA1 Plus", "controller": "Tour Box Elite", "light control": "Elgato Stream Deck Plus", "tablet": "iPad Pro", "tablet arm": "Cooks you desk arm", "monitor mount": "BenQ monitor mount", "travel charger": "ESR travel charger", "desk mat": "Grovemade felt mat", "smart home device": "Amazon Alexa Show", "security cameras": "UV security cameras", "Mac Mini": "Mac Mini Pro"}
                ```
            """
            },
            {
                "role":"user",
                "content": text
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0]['message']['content']

# since openai api have rate limit of 3 request per minute, we will sleep 20s for each request
import time

# for index, row in df.iterrows():
#     if index % 3 == 0:
#         time.sleep(20)
#     df.loc[index, 'items'] = openai_api(row['transcript_text'])

#df['items'] = df['transcript_text'].apply(openai_api)

# save results to csv
#df.to_csv('youtube-desk-setup.csv', index=False)


import requests
hf_api_key = os.environ.get("HF_API_KEY")


API_URL = "https://api-inference.huggingface.co/models/distilbert-base-cased-distilled-squad"
headers = {"Authorization": f"Bearer {hf_api_key}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	

# for index, row in df.iterrows():
#     ans = {"answer": "NA"}
#     try: 
#         ans = query({
#         "inputs": {
#             "question": "What is Operating System used? Windows, MacOS or Linux?",
#             "context": row['transcript_text']
#         }
#     })
#     except:
#         pass
#     df.loc[index, 'OS'] = ans['answer']
#     print(f"row : {index} done")
#     print(ans)
    # if index >= 300:
    #     break
#print(df['OS'].value_counts())
#df.to_csv('./data/after-LLM-data.csv', index=False)
# output = query({
# 	"inputs": {
# 		"question": "What is my name?",
# 		"context": "My name is Clara and I live in Berkeley."
# 	},
# })
