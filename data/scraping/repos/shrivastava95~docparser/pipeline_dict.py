import clip
import torch
from pytesseract import pytesseract, Output
from PIL import Image
import os
from .data import lang_mapping
from .pipeline_clip import *
import openai
import html
from time import time, sleep
from io import StringIO
import pandas as pd


separator = '|'


def chunk_total_text(total_text, word_limit):
    lines = total_text.split('\n')
    chunks = []

    for line in lines:
        if len(chunks) == 0:
            chunks.append(line)
        elif len(chunks[-1]) + len(line) < word_limit:
            chunks[-1] = chunks[-1] + '\n' + line
        else:
            chunks.append(line)
    return chunks
    

def prompt_template(chunk):
    # input_text_decoded = html.unescape(chunk.read().decode('utf-8'))
    input_text_decoded = chunk
    messages = [
        {
            "role": "system", 
            "content": "You will help me parse raw noisy OCR output into neat tables. The output should contain the proper Odia characters instead of escape sequences."
        },
        {
            "role": "user", 
            "content": """ANSWER WITH AS LITTLE UNNECESSARY TEXT AS POSSIBLE (just the table. no need to describe anything about it at the start.). The table should have three columns:
1. Word in English
2. Part of speech it belongs to
3. Translation of the word

Here is the raw data. Format it into the table. For example, here are the first two rows of the table:

| a | Adjective&Article   | ଅରୋଟ୍‌ |
| accordingly |	Adverb  |  ଆଦିଙ୍କ୍‌ ଲେକେ |"""
        },
        {
            "role": "assistant",
            "content": "Sure! Please provide me with the raw data, and I'll format it into a three-column table for you."
        },
        {
            "role": "user",
            "content": input_text_decoded
        }
    ]
    return messages


def query_chatgpt_until_complete(messages, temperature, model):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                max_tokens=2000,
                messages=messages
            )
            gpt_text = html.unescape(completion.choices[0].message['content'])
            break
        except Exception as e:
            # print(e)
            sleep(5)
            continue
    return gpt_text


def clean_dataframes(dataframes):
    cleaned_dfs = []
    for df in dataframes:
        try:
            if df.shape[1] != 5:
                continue
            df = df.iloc[:, 1:-1]  # remove blank unused columns
            df.columns = ['English Word', 'Part of Speech', 'Translation']
            cleaned_dfs.append(df)
        except Exception as e:
            # print(e)
            pass
    return cleaned_dfs    


# dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text'])
def image_to_dict(model, preprocess, device, image, lang, config, mode, openai_api_key=None, gpt_model='gpt-3.5-turbo'):
    assert openai_api_key is not None
    total_text = image_to_string_v2(model, preprocess, device, image, lang, config, mode=mode)
    chunks = chunk_total_text(total_text, word_limit=1500)
    gpt_outputs = []
    for chunk in chunks:
        messages = prompt_template(chunk)
        openai.api_key = openai_api_key
        gpt_text = query_chatgpt_until_complete(messages, 0.4, gpt_model)
        gpt_outputs.append(gpt_text)
    
    dataframes = [
        pd.read_csv(StringIO(gpt_text), 
                    sep=separator, 
                    on_bad_lines='skip')
        for gpt_text in gpt_outputs
    ]

    cleaned_dfs = clean_dataframes(dataframes)
    data = pd.concat(cleaned_dfs, axis=0).reset_index(drop=True)
    # print(data.shape)
    # print(data.head())

    return data