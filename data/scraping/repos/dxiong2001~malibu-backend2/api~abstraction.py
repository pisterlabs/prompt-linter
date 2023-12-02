import json
import requests
import urllib
import urllib.parse
from decouple import config
import time
import os
import openai

# from transformers import pipeline

# #classifier = pipeline("summarization", model="t5-base")


# def abs_summarization(article_sections):
#     summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="tf")
#     summ_sections = []
#     for text in article_sections:
#         summ_sections.append(summarizer(text, max_length = 100, min_length = 10, do_sample=False)[0]['summary_text'])
#     return summ_sections



def abs_summarization(article_sections):
    start_time = time.time()
    summ_sections = []
    for text in article_sections:
        
    
        headers = {
        "Authorization": f"Bearer {config('HUGGING_FACE')}",
        }
        
        data = json.dumps(text)
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        response = requests.request("POST", API_URL, headers=headers, data=data)
        print(json.loads(response.content.decode("utf-8")))
        abs_summ = json.loads(response.content.decode("utf-8"))[0]['summary_text']
        
        print(abs_summ)
        summ_sections.append(abs_summ)
    print("--- %s seconds ---" % (time.time() - start_time))
    return summ_sections

def process_text(text):
    split_text = text.split(".")
    return_text = ""
    for split in split_text:
        print(split)
        if(split!=" " or split!="." or split!="  "):
            return_text = return_text + split.strip() + ". "
        
    return_text = return_text.replace("\n","")
    return return_text.strip()


def abs_summarization2(article_sections):

    openai.api_key = os.getenv("OPENAI_API_KEY")
    abs_summary_sections = []
    
    for section in article_sections:
        
        #use try/except in case openai server goes down
        try:
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt = "Summarize this for a second-grade student:\n\n"+section,
                temperature=0.7,
                max_tokens=64,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            print(response)
            abs_summary_sections.append(process_text(response['choices'][0]['text']))
        except:
            abs_summary_sections.append(".")
    return abs_summary_sections

    