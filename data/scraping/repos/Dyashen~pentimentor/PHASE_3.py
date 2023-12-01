from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import spacy
from langdetect import detect
import pandas as pd
import requests, spacy, os
from langdetect import detect
from googletrans import Translator
from deep_translator import GoogleTranslator
import openai
import numpy as np


folder_path = 'scripts\pdf'
dutch_spacy_model = "nl_core_news_md"
english_spacy_model = "en_core_web_sm"

dict = {
    'nl':'nl_core_news_md',
    'en':'en_core_web_sm'
}

total_df = None
gt = Translator()

huggingfacemodels = {
    'T1':"https://api-inference.huggingface.co/models/haining/scientific_abstract_simplification",
    'T2': "https://api-inference.huggingface.co/models/sambydlo/bart-large-scientific-lay-summarisation",
    'T3': "https://api-inference.huggingface.co/models/philippelaban/keep_it_simple"
}

max_length = 2000
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

languages = {
    'nl':'nl_core_news_md',
    'en':'en_core_web_md'
}

class HuggingFaceModels:
    def __init__(self, key=None):
        global huggingface_api_key
        try:
            huggingface_api_key = key
        except:
            huggingface_api_key = 'not_submitted'

    """"""
    def query(self, payload, API_URL):
        headers = {"Authorization": f"Bearer {huggingface_api_key}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    """"""
    def scientific_simplify(self, text, lm_key):
        try:
            API_URL = huggingfacemodels.get(lm_key)
            translated = GoogleTranslator(source='auto', target='en').translate(str(text))
            
            if lm_key == 'T1':
                result = self.query({"inputs": str('simplify: ' + str(translated)),"parameters": {"max_length": len(sentence)+10},"options":{"wait_for_model":True}}, API_URL)
            else:
                result  = self.query({"inputs": str(translated),"parameters": {"max_length": len(sentence)+10},"options":{"wait_for_model":True}}, API_URL)
            
            if 'generated_text' in result[0]:
                translated = GoogleTranslator(source='auto', target='nl').translate(str(result[0]['generated_text']))
                return translated
            elif 'summary_text' in result[0]:
                translated = GoogleTranslator(source='auto', target='nl').translate(str(result[0]['summary_text']))
                return translated
            else:
                return None
        except:
            return text

def tokenize_text(text):
    txt_language = detect(text)
    dic_language = languages.get(txt_language)
    nlp = spacy.load(dic_language)
    doc = nlp(text)
    return doc.sents

def process_file(file_path):
    with open(folder_path + '/' + file_path, "r", encoding='utf8') as file:
        text = file.read()
        tokens = tokenize_text(text)
        sentences = []
        for s in tokens:
            try:
                sentences.append(s)
            except Exception as e:
                print(e)
        
        sentences = np.array(sentences)
        pad_size = 5 - (sentences.size % 5)
        padded_a = np.pad(sentences, (0, pad_size), mode='empty')
        paragraphs = padded_a.reshape(-1, 5)
        return paragraphs

hf = HuggingFaceModels(os.getenv('huggingface-api-key'))
original_scientific_papers = [f for f in os.listdir(folder_path)]

for paper in original_scientific_papers:
    sentence_tokens = process_file(paper) 
    for sentence in sentence_tokens:
        for model in huggingfacemodels.keys():
            filename = "SIMPLIFIED_"+model+'_'+paper
            with open(filename, 'a', encoding='utf-8') as f:
                output = hf.scientific_simplify(str(sentence), model)
                f.write(str(output)) 