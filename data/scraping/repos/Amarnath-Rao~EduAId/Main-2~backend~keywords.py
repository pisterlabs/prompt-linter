import io
from flask import Flask, jsonify, request, json, send_from_directory
from google.cloud import texttospeech as tts
import tkinter
from tkinter import filedialog
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from collections import Counter
import pprint
import google.generativeai as palm
from dotenv import load_dotenv
import os
load_dotenv()
from flask import Flask, jsonify, request, json, send_from_directory
import os
from google.cloud import texttospeech as tts
import openai
import pandas as pd
import base64
import os 
import requests
import numpy as np
import pprint
import google.generativeai as palm


from dotenv import load_dotenv
load_dotenv()

from flask_cors import CORS
palm.configure(api_key=os.getenv("PALM_API_KEY"))







root = tkinter.Tk()

# Hide unnecessary GUI element
root.withdraw()
filename = filedialog.askopenfilename()
print(filename)
print('Processing file...')

# Edit keyword_num to set the number of keywords
keyword_num = 20

resource_manager = PDFResourceManager()
fake_file_handle = io.StringIO()
converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
page_interpreter = PDFPageInterpreter(resource_manager, converter)

with open(filename, 'rb') as file:
    for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
        page_interpreter.process_page(page)

    text = fake_file_handle.getvalue()

converter.close()
fake_file_handle.close()

words = text.lower().replace('\n', ' ').split(' ')

words = list(filter(''.__ne__, words))
stopwords = set(open(str(os.path.dirname(os.path.abspath(__file__))) + '/stopwords_english', 'r').read().splitlines())
filtered_words = []





def generate_story(new_array: list) -> str:
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name

    # Combine all words into a single prompt
    keyword_str = ', '.join(keywords)
    prompt = f"Generate a comprehensive story on the topic related to {keyword_str} that helps students for last-minute revision. Provide information about key concepts, applications, and significance"

    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1024,
    )

    content = completion.result
    content = content.encode().decode('unicode_escape')
    title = content.split('\n')[0]
    title = title.replace('Title: ', '')
    story = content[content.find('\n'):]
    story = story.lstrip()

    return title, story

# filtering words such that they don't repeat
def filter_function(x):
    if x not in stopwords:
        filtered_words.append(x)

list(map(filter_function, words))
word_frequency = Counter(filtered_words)
keywords = [word for word, _ in word_frequency.most_common(keyword_num)]

# Store keywords in a new array
new_array = keywords
title, story = generate_story(new_array)
print(f"Title: {title}")
print(f"Story: {story}")


def text_to_wav(text: str, title, dest, voice_name = "en-IN-Wavenet-A"):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16,
                                   speaking_rate=0.8)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )
    
    filename = f"{dest + '/' + title.translate(str.maketrans('', '', '*'))}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

    return filename
