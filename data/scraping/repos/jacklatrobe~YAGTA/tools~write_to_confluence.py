import os
import re
import base64
import requests
import json
import logging
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.agents import Tool
from nltk import download as nltk_download
from nltk.tokenize import word_tokenize
nltk_download('stopwords')
nltk_download('punkt')
from nltk.corpus import stopwords

# Check for required ENV variables
if(os.environ.get("CONFLUENCE_CLIENT_ID") == None):
    msg = "CONFLUENCE_CLIENT_ID not set as env variable"
    logging.error(msg)
    raise ValueError(msg)

if(os.environ.get("CONFLUENCE_API_KEY") == None):
    msg = "CONFLUENCE_API_KEY not set as env variable"
    logging.error(msg)
    raise ValueError(msg)

if(os.environ.get("CONFLUENCE_SITE_NAME") == None):
    msg = "CONFLUENCE_SITE_NAME not set as env variable"
    logging.error(msg)
    raise ValueError(msg)

# Splits strings into chunks of X length
def split_str(seq, chunk, skip_tail=False):
    lst = []
    if chunk <= len(seq):
        lst.extend([seq[:chunk]])
        lst.extend(split_str(seq[chunk:], chunk, skip_tail))
    elif not skip_tail and seq:
        lst.extend([seq])
    return lst
    
# The function that the tool executes - this one writes to confluence
def write_to_confluence(text) -> str:
    # Replace these variables with your own values
    atlassian_email = os.environ.get("CONFLUENCE_CLIENT_ID")
    api_token = os.environ.get("CONFLUENCE_API_KEY")
    your_site_name = os.environ.get("CONFLUENCE_SITE_NAME")

    # Encode the email and API token in base64 format
    credentials = "{atlassian_email}:{api_token}".format(atlassian_email=atlassian_email, api_token=api_token)
    encoded_credentials = base64.b64encode(bytes(credentials, 'UTF-8')).decode()

    # Create a title
    summary_llm = OpenAI(max_tokens=128)
    title = summary_llm("What is a good title for a page containing this text: {text}".format(text=text))

    expansion_llm = OpenAI(max_tokens=2000)
    if len(text) < 1000:
        text = expansion_llm("Re-write and expand this text into a full page article, creating additional headings with information where necessary, and format it as body HTML for a confluence page: {text}".format(text=text))
    else:
        text = expansion_llm("Format this text as HTML, with headings, lists and paragraphs: {text}".format(text=text))

    # Set the base URL and the search query parameters
    base_url = f"https://{your_site_name}.atlassian.net/wiki/api"
    create_endpoint = "/v2/pages"
    
    # Set the headers, including the encoded credentials for authorization
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Basic {encoded_credentials}".format(encoded_credentials=encoded_credentials)
    }

    # Make the POST request to the Confluence API
    payload = json.dumps( {
        "spaceId": 26378243,
        "status": "current",
        "title": title,
        "body": {
            "representation": "storage",
            "value": text
        }
        } )
    response = requests.request(
        "POST",
        base_url + create_endpoint,
        data=payload,
        headers=headers
    )

    # Check if the request was successful (HTTP status code 200) and build YAML response obj
    if response.status_code == 200:
        return "Article was successfully created: {title}".format(title=title)
    else:
        print(response.text)
        return "Unable to save article. Request failed with status code {response_code}".format(response_code=response.status_code)
    
# Define the tool as LangChain tool objects for agent use
WriteToConfluenceTool = Tool(
    name = "Write to confluence",
    func = write_to_confluence,
    description="Useful for saving new pages to the users private confluence knowledgebase. Input must be a full page of text"
)