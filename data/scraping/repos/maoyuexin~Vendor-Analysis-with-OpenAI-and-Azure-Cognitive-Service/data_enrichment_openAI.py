import urllib.request
import json
import os
import argparse
from pathlib import Path
import openai
import re


parser = argparse.ArgumentParser()
parser.add_argument("--file_path_in", type=str)
parser.add_argument("--gpt_engine", type=str)
parser.add_argument("--temperature_clasify", type=float, default=0.8)
parser.add_argument("--max_tokens_clasify", type=int, default=100)
parser.add_argument("--temperature_summarize", type=float, default=0.4)
parser.add_argument("--max_token_summarize", type=int, default=1024)

parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--frequency_penalty", type=float, default=0.5)
parser.add_argument("--presence_penalty", type=float, default=0.5)
parser.add_argument("--best_of", type=float, default=1)

parser.add_argument('--file_path_out', type=str)



openai.api_type = "azure"
openai.api_key = '<add api_key>'
openai.api_base = '<add api_base>'
openai.api_version = "<add api_version>"
entity_types = ["Research Orgnization","Technology","Tobacco" , "Media"]


args = parser.parse_args()

def splitter(n, s):
    pieces = s.split()
    list_out = [" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n)]
    return list_out

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def trim_incomplete(t):
    if t.endswith('.'):
        if not re.search('[a-z]\.$',t):
            t = t[:-1]

    if not t.endswith('.'):
        t = t.rsplit('. ', 1)[:-1]
        t = "".join(t)+'.'
    
    t = t.strip()
    return t


def recognize_categories(prompt, engine,  temperature, max_tokens):
    """Recognize entities in text using OpenAI's text classification API."""
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].text



def summarize_information(prompt, engine, temperature , max_tokens , top_p , frequency_penalty , presence_penalty , best_of):
     
    response = openai.Completion.create(
        engine= engine,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty =presence_penalty,
        stop=['\n\n###\n\n'], #the ending token used during inference, once reaches this token GPT-3 knows the completion is over
        best_of = best_of
    )
    return response.choices[0].text



output_path = f"{Path(args.file_path_out)}/categorized_json/"
print(f"output_path{output_path}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

path = f"{Path(args.file_path_in)}/enriched_json/"

files = os.listdir(path)


print(f"temperature_clasify:{args.gpt_engine}")
print(f"max_tokens_clasify:{args.temperature_clasify}")
print(f"frequency_penalty:{args.frequency_penalty}")



for fileName in files:
    if fileName not in ["tracker.csv"]:
        print(f"{path}{fileName}")
        with open(f"{path}{fileName}", 'rb') as f2: 
            json_results = json.load(f2)
        
        for idx, invoice in enumerate(json_results['documents']): 
            vendorName = invoice.get("VendorName")
            vendorDescription = invoice.get("Bing_Search_Description")


            #### Summarize vendor information 
            try:
                
                prompt_i = 'Summarize the vendor introduction, industry and products given the vendor name.\n\nvendor name:\n'+normalize_text(str(vendorName))+'\n\nSummary:\n'    

                #if vendorDescription:
                #    prompt_i = 'Summarize the vendor information given the vendor name and use the text for reference\n\nvendor name:\n'+normalize_text(str(vendorName))+ '\n\nText:\n'+normalize_text(vendorDescription)+'\n\nSummary:\n'
                #else:
                #    prompt_i = 'Summarize the vendor introduction given the vendor name.\n\nvendor name:\n'+normalize_text(str(vendorName))+'\n\nSummary:\n'

                vendor_info = normalize_text(summarize_information(normalize_text(prompt_i), engine=args.gpt_engine, temperature=args.temperature_clasify, max_tokens=args.max_tokens_clasify, top_p=args.top_p, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty, best_of=args.best_of))
                    
            except:
                vendor_info = ""

            #### Vendor Categorization
             
            if vendor_info =="":
                category = 'Not Identified'
            else:
                try:

                    prompt = f"Classify the following vendor information into 1 of the following categories: : {', '.join(entity_types)} \n\nvendor information:{vendor_info}\n\nClassified category:"
                    category = normalize_text(recognize_categories(prompt, engine=args.gpt_engine,  temperature=args.temperature_clasify, max_tokens=args.max_tokens_clasify))
                except:
                    category = 'Not Identified'
            print(f"{category}:{vendor_info}")
            json_results['documents'][idx]['Data_Category'] = category
            json_results['documents'][idx]['Data_Summarization'] = vendor_info
        
        with open(f"{output_path}{fileName}", 'w') as f3: 
            json.dump(json_results, f3)

print(f"OpenAI data categorzation process complete")