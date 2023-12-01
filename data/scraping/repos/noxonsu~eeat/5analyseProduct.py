#4analyseProduct.py
import json
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain.schema import SystemMessage, HumanMessage
import os
from langchain.chat_models import ChatOpenAI
from bs4 import BeautifulSoup
import re
import time
import hashlib
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
import json
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
import openai
import os

from utils import *

INDUSTRY_KEYWORD = os.environ.get('INDUSTRY_KEYWORD')
ADDITIONAL_INFO = os.environ.get('ADDITIONAL_INFO','')
SERP_PRICES_EXT = os.getenv('SERP_PRICES_EXT')
BASE_GPTV = os.environ.get('BASE_GPTV','gpt-3.5-turbo-1106')
SMART_GPTV = os.environ.get('SMART_GPTV','gpt-3.5-turbo-1106')

cfl = load_from_json_file("7key_features_optimized.json","data/"+INDUSTRY_KEYWORD)
# Check if exists.
if cfl:
    clusterized_features_list_f = "Key features according to optimized feature list: "+cfl['title']+": \n "+json.dumps(cfl['features'])
else:
    clusterized_features_list_f = "Key features ('key_features' in JSON))"


def load_summaries_from_data_folder(folder_path="data"):
    """Loads summaries from all JSON files in the specified folder."""
    summaries = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                content = json.load(f)
                if "summary" in content:
                    domain_name = filename.rstrip(".json")  # Extract domain name from filename
                    summaries[domain_name] = content["summary"]
    return summaries




prompt = """Instructions for Assistant to Analyze """+INDUSTRY_KEYWORD+""" Products

Objective: To determine the critical characteristics or features of the chosen """+INDUSTRY_KEYWORD+""" that may be important to consumers.

Preparation:
For each service, list down how they fare in each of the criteria.

In additional detrmine such information
1. Call to action - 'talk to a manager', 'book a demo', 'talk to team', sign up etc.
2. Their usecases  
3. Their solutions. 
4. """+clusterized_features_list_f+""""
5. """+ADDITIONAL_INFO+"""?
                                                                
The goal is to provide an objective view of each product offerings, highlighting both strengths and potential areas for improvement. Provide the results in JSON format."""


def get_hash(prompt):
    # That a md5 hash
    return hashlib.md5(prompt.encode()).hexdigest()

prompt_hash = get_hash(prompt)

# Write pond into the file
with open("data/"+INDUSTRY_KEYWORD+"/5prompt_"+prompt_hash+".txt", "w") as file:
    file.write(prompt)

def get_company_details(company):
    """Extract details of the company using the LLMChain."""
  
    summary=load_from_json_file(company+".json","data/"+INDUSTRY_KEYWORD)
    question_content = 'INDUSTRY_KEYWORD: '+INDUSTRY_KEYWORD+"\n\n"+json.dumps(summary['summary'])
    question_content = question_content[:40000]

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=question_content)
    ]
    start = time.time()
    print(SMART_GPTV)
    response = openai.ChatCompletion.create(
            temperature=0,
            model=SMART_GPTV,  # Update this to the model you're using
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question_content}
            ]
        )

    if response['choices'][0]['message']['content']:
        json1 = json.loads(response['choices'][0]['message']['content'])
    else:
        json1 = "Not found"
    end = time.time()
    print("Time to get response1: "+str(end - start))
    try:
        start = time.time()
        response = openai.ChatCompletion.create(
                model=SMART_GPTV,  # Update this to the model you're using
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "Find "+SERP_PRICES_EXT+" and determin business model. status 'Not found' if not found or error. Return JSON with status and 'priceAndPlans'."},
                    {"role": "user", "content": json.dumps(summary['priceAndPlans'])}
                ]
            )
        
        ch = response['choices'][0]['message']['content']
        json2 = json.loads(ch)


        end = time.time()
        print("Time to get response2: "+str(end - start))


    
        
        #add prices and plans
        json1['pricesAndPlans'] = json2
        return json1
    except Exception as e:
        print(f"Failed to decode JSON for response: {e}")
        return None
    

def main():
    
    summaries = load_from_json_file("1companies.json", "data/" + INDUSTRY_KEYWORD)
    
    # Filter the summaries to get only those with nature "single project"
    filtered_summaries = {k: v for k, v in summaries.items() if v.get('nature') == "single project" and v.get('5prompt_Hash') != prompt_hash and v.get('priceAndPlansCrawled') != 'Not found'}
    total=len(filtered_summaries)
    print (total)
    # Load existing company details
    company_details = load_from_json_file("5companies_details.json", "data/" + INDUSTRY_KEYWORD)
    
    # Iterate through the filtered summaries to get details for each company
    i = 0
    for company, compdata in filtered_summaries.items():
        i += 1
        print(f"{(i / total) * 100}%")
        print(company)

        print(f"Analysing details for company: {company}")
        
        details = get_company_details(company)

        if details:
            company_details[company] = details
            save_to_json_file(company_details, "5companies_details.json", "data/" + INDUSTRY_KEYWORD)
            summaries[company]['5prompt_Hash'] = prompt_hash
            save_to_json_file(summaries, "1companies.json", "data/" + INDUSTRY_KEYWORD)


if __name__ == "__main__":
    main()
    if not cfl:
        print("No optimized features found. Please run 6ClusterAndOptimizeFeatures after")

