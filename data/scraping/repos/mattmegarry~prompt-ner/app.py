import sys
import streamlit as st
import anthropic
from config import ANTHROPIC_API_KEY
import json


claude = anthropic.Client(ANTHROPIC_API_KEY)

st.title("Constituency Selected: Coventry South")
st.write("Ask a question about 517 tweets by the MP for this area, Zarah Sultana.")

with open('sultana_517.json') as f:
  data = json.load(f)

all_tweets = ''
for item in data:
    all_tweets += item['text']

prompt = st.text_input('From the tweets I have provided, please give me a list of:')

if prompt:
    prompt = f"{anthropic.HUMAN_PROMPT}{all_tweets}\n\nFrom the tweets I have provided, please give me a list of {prompt}{anthropic.AI_PROMPT}"
    print(prompt)
    response = claude.completion(prompt=prompt, model="claude-2", max_tokens_to_sample=2000)
    print(response)
    print(type(response))
    st.write(response['completion'])

    


