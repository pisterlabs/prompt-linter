import platform
import os

import openai

import re
from dotenv import load_dotenv
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
load_dotenv()
print('Python: ', platform.python_version())
print('re: ', re.__version__)
print('nltk: ', nltk.__version__)

import time
start_time = time.time()


def count_tokens(filename):
    with open(filename, 'r') as f:
        text = f.read()
        
    tokens = word_tokenize(text)
    num_tokens = len(tokens)
    return num_tokens

filename = "sample_meeting.txt"
token_count = count_tokens(filename)
print(f"Number of tokens: {token_count}")


def break_up_file(tokens, chunk_size, overlap_size):
    if len(tokens) <= chunk_size:
        yield tokens
    else:
        chunk = tokens[:chunk_size]
        yield chunk
        yield from break_up_file(tokens[chunk_size-overlap_size:], chunk_size, overlap_size)

def break_up_file_to_chunks(filename, chunk_size=2000, overlap_size=100):
    with open(filename, 'r') as f:
        text = f.read()
    tokens = word_tokenize(text)
    return list(break_up_file(tokens, chunk_size, overlap_size))

# filename = "sample_meeting.txt"

chunks = break_up_file_to_chunks(filename)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} tokens")

if chunks[0][-100:] == chunks[1][:100]:
    print('Overlap is Good')
else:
    print('Overlap is Not Good')

openai.api_key = os.getenv("api_key")

def convert_to_prompt_text(tokenized_text):
    prompt_text = " ".join(tokenized_text)
    prompt_text = prompt_text.replace(" 's", "'s")
    return prompt_text

response = []
prompt_response = []

chunks = break_up_file_to_chunks(filename)
for i, chunk in enumerate(chunks):
    prompt_request = "Summarize this meeting transcript: " + convert_to_prompt_text(chunks[i])

    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_request,
            temperature=.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
    )

    prompt_response.append(response["choices"][0]["text"])

prompt_request = "Consoloidate these meeting summaries: " + str(prompt_response)

response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_request,
        temperature=.5,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

filename = "sample_meeting.txt"

action_response = []

chunks = break_up_file_to_chunks(filename)
for i, chunk in enumerate(chunks):
    prompt_request = "Provide a list of action items with a due date from the provided meeting transcript text: " + convert_to_prompt_text(chunks[i])

    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_request,
            temperature=.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
    )
    
    action_response.append(response["choices"][0]["text"])

print(action_response)

prompt_request = "Consoloidate these meeting action items: " + str(action_response)

response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_request,
        temperature=.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

meeting_action_items = response["choices"][0]["text"]
print(meeting_action_items)

print("Time taken",time.time() - start_time)