#5visual.py.py
import json
import time
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain.schema import SystemMessage, HumanMessage
import os
from langchain.chat_models import ChatOpenAI
from bs4 import BeautifulSoup
import re
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

from utils import *
BASE_GPTV = os.environ.get('BASE_GPTV','gpt-3.5-turbo-1106')
SMART_GPTV = os.environ.get('SMART_GPTV','gpt-3.5-turbo-1106')

# Function to check if existing domains are included in the output
def check_domains_in_output(existedDomainList, output, article_index):
    for existedDomain in existedDomainList:
        if existedDomain.lower().replace("www.","") not in output.lower():
            print(f"{existedDomain} not in output of this file article{article_index}.md")
            exit()


# Environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
INDUSTRY_KEYWORD = os.environ.get('INDUSTRY_KEYWORD', 'Vector databases')

# Load clusterized features list
cfl = load_from_json_file("7key_features_optimized.json", "data/" + INDUSTRY_KEYWORD)
if cfl:
    clusterized_features_list_f = "features list for projects (detect which aplyable to the project): " + cfl['title'] + ": \n " + cfl['intro'] + json.dumps(cfl['features'])
else:
    clusterized_features_list_f = ""
#old Write a final SEO optimized article about [INDUSTRY_KEYWORD] using the data. Compare all the elements and find the best one project.  Need an article with tables, etc. Return as Markdown with Title, Meta keywords, Meta description, Text fields (without ")

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content="""Act like an analytic. Need to create a comparsion article across the [INDUSTRY_KEYWORD] list. 
I will send you projects one by one. Analyse every product, then add given information to main article. 
What prices they have? What should i do to start using? Focus on differencies between porojects.
Return only main article every time we send you new project. Use tables and other markdown syntaxis.
Keep domain names in article.
   """
.replace("[INDUSTRY_KEYWORD]", INDUSTRY_KEYWORD)),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)


if __name__ == "__main__":
    # Load data from the /data folder
    all_sites_data = load_from_json_file("5companies_details.json", "data/" + INDUSTRY_KEYWORD)
    sites = list(all_sites_data.keys())
    existedDomainList = []
    datatoAdd = {}
    datatoAddFeatures = {}
    for i, domain in enumerate(sites):
        datatoAdd[domain] = all_sites_data[domain]['pricesAndPlans']
        if ('features' not in all_sites_data[domain]):
            print("key_features not in "+domain)
            continue
        if ('features' in all_sites_data[domain]):
            all_sites_data[domain]['key_features'] = all_sites_data[domain]['features']
        datatoAddFeatures[domain] = all_sites_data[domain]['key_features']



    messages = [
        SystemMessage(content="Act like an analytic. Compare pricing plans and create comparsion review of "+INDUSTRY_KEYWORD+". Don't include projects without numbers. Keep company names and features. Use tables if possible. Return markdown."),
        HumanMessage(content=json.dumps(datatoAdd)+json.dumps(datatoAddFeatures))
    ]

    start = time.time()
    try:
        mod = SMART_GPTV #gpt-4-1106-preview
        chat = ChatOpenAI(temperature=0, model_name=mod)
        response = chat(messages)
        print("Pricing using "+mod)
    except Exception as e:
        raise Exception("Failed to get response from")
        
    
    end = time.time()

    print("Time to get response1: "+str(end - start))

    
    with open(f"data/{INDUSTRY_KEYWORD}/article_pricing.md", "w") as f:
                f.write(response.content)
    with open(f"data/{INDUSTRY_KEYWORD}/article_pricing_inpout.md", "w") as f:
                f.write(json.dumps(datatoAdd))


print("8visual done \n\n ")