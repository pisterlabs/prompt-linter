import os
from dotenv import load_dotenv

from langchain.chat_models import ChatGooglePalm
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate

import requests
import json
from bs4 import BeautifulSoup

load_dotenv(dotenv_path="./key.env")
serper_api_key = os.getenv("SERPER_API_KEY")
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
palm_key = os.getenv("PALM_KEY")

# Search Tools
def search(query):
    url = "https://google.serper.dev/search"
    
    # Query
    payload = json.dumps({
        "q": query
    })
    
    # Header
    header = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=header, data=payload)
    print(response.text)
    return response.text

# search("What is the capital of the United States?")

def summarise(objective, content):
    llm = ChatGooglePalm(google_api_key = palm_key, temperature = 0, model_name = "models/chat-bison-001")
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n"], chunk_size = 10000, chunk_overlap = 500)
    docs = text_splitter.create_documents([content])
    
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    Summary:
    """
    
    map_prompt_template = PromptTemplate(
        template = map_prompt, input_variables = ["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm = llm,
        chain_type = 'map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = True
    )
    
    output = summary_chain.run(input_documents = docs, objective = objective)
    return output


# Scrape Tool
def scrape(objective: str, url: str):
    print("Scraping...")
    
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    
    data = {
        "url": url
    }
    
    data_json = json.dumps(data)
    
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers = headers, data = data_json)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        print(f"Content: {text}")
        print(len(text))
        output = summarise(objective, text)
        return output

    else:
        print(f"Error: {response.status_code}")
    
# scrape("What was the Battle of the Bulge?", "https://history.army.mil/html/reference/bulge/index.html")
