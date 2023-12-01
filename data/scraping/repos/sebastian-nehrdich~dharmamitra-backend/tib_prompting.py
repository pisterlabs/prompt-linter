import pandas as pd
import re
import requests
import pyewts
from indexing import TibIndex
from prompting import OpenAiChatGPT4

dictionary_path = "data/tib_dictionary.tsv"

OpenAiModel = OpenAiChatGPT4()

Index = TibIndex()

pyewts = pyewts.pyewts()


def create_dictionary(path):    
    # read dictionary from path csv
    df = pd.read_csv(path, sep='\t', encoding='utf-8')        
    df = df.dropna(subset=['english1'])
    # creater dictionary with column 'word' as key and column 'meanings' as value
    dictionary = {}
    for index, row in df.iterrows():
        current_head_word = row['definiendum']        
        if len(current_head_word) >= 8:
            if current_head_word not in dictionary:
                dictionary[current_head_word] =  str(row['english1']).strip()
            else:
                dictionary[current_head_word] += "; " + str(row['english1']).strip()
    return dictionary
    

tib_dictionary = create_dictionary(dictionary_path)

def translate(query):    
    query = query.replace(" |", "|")
    query = query.replace("|", "/")
    query = pyewts.toUnicode(query)
    # run POST request to translate Tibetan to English to localhost:3401 with {input_sentence: query}
    response = requests.post("http://localhost:3401/translate-tib/", json={"input_sentence": query})
    # read response as json    
    response = response.text    
    return response

def preprocess_query(query):
    query = query.replace(" |", "|")
    query = query.replace("||", "/_/")
    query = query.replace("|", "/")    
    query = query.replace(" /", "/")    
    query = query.replace("\n", " ")
    return query


def create_dictionary_entries(tib_string):
    result = ""
    for key in tib_dictionary:
        if key in tib_string:
            result += key + " means " + tib_dictionary[key.replace("\n", " ")] + "\n"
    return result

def create_samples(query):
    results = Index.search(query, 20)
    result_string = ""
    for result in results:
        result_string += "Tibetan " + result[0] + " means English: " + result[1] + "\n"
    return result_string

def construct_prompt(query, explanation_level):    
    prompt = ""
    query = preprocess_query(query)
    prompt += "This machine translation might help you, but it might be faulty and need some postcorrection: " + translate(query) + "\n"
    prompt += "These example translations from English to Classical Tibetan will help you and are more important than the machine translation:\n " + create_samples(query) + "\n"
    prompt += "These dictionary entries might help you: " + create_dictionary_entries(query) + "\n"
    prompt += "\nTranslate the following Classical Tibetan into English: " + query + "\n"
    print(prompt)
    if explanation_level == 0:
        prompt += "Output only the English translation.\n"
    elif explanation_level == 1:
        prompt += "Output only the English translation together with a brief explanation of the meaning and function of the Tibetan words.\n"
    elif explanation_level == 2:
        prompt += "Output only the English translation together with a brief and concise explanation of its structure and the meaning of the Tibetan words.\n"    
    return prompt


def get_translation_tib(input_sentence, explanation_level=0):
    if not re.search(r'[a-z][a-z][a-z]', input_sentence):
        input_sentence = pyewts.toWylie(input_sentence)
    prompt = construct_prompt(input_sentence, explanation_level)
    response = OpenAiModel.get_translation(prompt)
    return response




