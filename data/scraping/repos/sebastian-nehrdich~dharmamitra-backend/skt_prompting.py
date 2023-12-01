
import pandas as pd
import requests
import openai
from translator import SktTagger

from indexing import SktIndex
from prompting import OpenAiChatGPT3, OpenAiChatGPT4, OpenAiDavinci
from utils.skt_utils import spell_out_pos_tags, replace_tags, create_dictionary


dictionary_path = "data/marco_dict.tsv"

OpenAIModel = OpenAiChatGPT4()
Index = SktIndex()

def create_samples(query):
    results = Index.search(query, 15)
    result_string = ""
    for result in results:
        result_string += "Sanskrit " + result[0] + " means English: " + result[1] + "\n"
    return result_string

skt_dictionary = create_dictionary(dictionary_path)

def postprocess_translation(translation):
    translation_items = translation.split(" # ")
    result = ""
    for item in translation_items:
        entries = item.split(" ")        
        tags = " ".join(entries[1:])
        tags = replace_tags(tags)
        entries = [entries[0]] + tags.split(" ")
        #entries = [entries[0]] + entries[2:]
        #if entries[0].strip() in skt_dictionary:
        #    entries.extend([" and possible meanings: " + skt_dictionary[entries[0].strip()]])
        result += " ".join(entries) + "\n"
    return result


def translate(query):    
    # tagger runs on a separate api that is called here  
    response = requests.post("http://localhost:3401/tag-sanskrit/", json={"input_sentence": query})
    response = response.text
    print("Response: ", response)
    return response

def construct_prompt(query, explanation_level):
    prompt = ""
    prompt += "These example translations from Sankrit to English might help you, but they are imprecise and need to be corrected according to the grammatical explanation that is supplied later:\n " + create_samples(query) + "\n"
    prompt += "This is a grammatical explanation of the sentences that you should use to create a correct translation. In case the spelling in the explanation is different to that of the main sentence, please follow the spelling of the main sentence. Stick very closely to this explanation and pay attention to the grammatical relationship of the words and the structure of the compounds: " + postprocess_translation(translate(query)) + "\n"
    prompt += "\nTranslate the following Sanskrit into English: " + query + "\n"
    print(prompt)
    if explanation_level == 0:
        prompt += "Output only the English translation.\n"
    elif explanation_level == 1:
        prompt += "Output only the English translation together with an explanation of the meaning of the Sanskrit words.\n"
    elif explanation_level == 2:
        prompt += "Output only the English translation together with an explanation of its structure and the meaning of the Sanskrit words.\n"    
    return prompt

def get_translation_skt(input_sentence, explanation_level=0):
    prompt = construct_prompt(input_sentence, explanation_level)
    response = OpenAIModel.get_translation(prompt)    
    return response


