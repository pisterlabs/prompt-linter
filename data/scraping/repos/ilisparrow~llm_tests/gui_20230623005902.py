## IMPORTS

from bs4 import BeautifulSoup              # Importing BeautifulSoup for HTML parsing
from bs4.element import Comment             # Importing Comment class for extracting comments from HTML
import urllib.request                       # Importing urllib.request for making HTTP requests
import streamlit as st                      # Importing streamlit for building interactive web apps
import os                                   # Importing os for accessing operating system functionalities
from dotenv import load_dotenv              # Importing load_dotenv for loading environment variables
from langchain.llms import OpenAI            # Importing OpenAI class from langchain.llms module
from langchain.prompts import PromptTemplate # Importing PromptTemplate class from langchain.prompts module
import json                                 # Importing json module for working with JSON data
from dotenv import dotenv_values            # Importing dotenv_values for loading environment variables from .env file
from googlesearch import search             # Importing search function from googlesearch module
import requests                            # Importing requests module for making HTTP requests
import unicodedata
import validators

## SETUP ENVIRONMENT VARIABLES

load_dotenv()
env_vars = dotenv_values(".env")



## Define system relevant input data for application
HARD_LIMIT_CHAR = 10000

## Functions

def tag_visible(element):
    excluded_tags = ['a', 'style', 'script', 'head', 'title', 'meta', '[document]']

    if element.parent.name in excluded_tags:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.find_all(text=tag_visible)
    visible_texts = [t.strip() for t in texts if t.strip()]

    return " ".join(visible_texts)


def extract_json_values(input_str):
    results = []
    while input_str:
        try:
            value = json.loads(input_str)
            input_str = ""
        except json.decoder.JSONDecodeError as exc:
            if str(exc).startswith("Expecting value"):   
                input_str = input_str[exc.pos+1:]
                continue
            elif str(exc).startswith("Extra data"):
                value = json.loads(input_str[:exc.pos])
                input_str = input_str[exc.pos:]
        results.append(value)
    return results

## Process website and save content to file
def process_website(url, output_file_name):
    html = urllib.request.urlopen(url).read()
    text_from_webpage = text_from_html(html)
    text_from_webpage = text_from_webpage[:HARD_LIMIT_CHAR]

    # Logging
    file_path = output_file_name
    with open(file_path, "w") as file:
        file.write(text_from_webpage)
    print("Variable content saved to the file:", file_path)
    return text_from_webpage

def get_link_based_on_article_name_via_google(article_title, url_to_watch):
    search = article_title + " " + url_to_watch
    url = 'https://www.google.com/search'

    headers = {
        'Accept' : '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82',
    }
    parameters = {'q': search}

    content = requests.get(url, headers = headers, params = parameters).text
    soup = BeautifulSoup(content, 'html.parser')

    search = soup.find(id = 'search')
    first_link = search.find('a')
    article_link= first_link['href']
    return first_link['href']


def prompt_to_llm_response(text_from_webpage, prompt_input):
    prompt = PromptTemplate(
        input_variables=["webpage", "prompt_text"],
        template="\"{prompt_text}\" \
            webpage :  \"{webpage}\"",
    )

    
    prompt_to_send = prompt.format(webpage=text_from_webpage, prompt_text=prompt_input)


    llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'], temperature=0)
    result_from_chatgpt = llm(prompt_to_send).replace("\n", "").replace("Answer:","")
    return result_from_chatgpt


def prompt_similarity_to_llm_response(sentence1, sentence2):
    prompt = PromptTemplate(
        input_variables=["sentence1", "sentence2"],
        template="""
            Compare the content of the following two sentences. Could sentence 1 be relevant for a person interested in sentence 2? 
            Answer with one of [strongly agree, agree, disagree, strongly disagree] only.

            Sentence 1: {sentence1}
            Sentence 2: {sentence2}
        """,
    )

    
    prompt_to_send = prompt.format(sentence1=sentence1, sentence2=sentence2)


    llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'], temperature=0)
    result_from_chatgpt = llm(prompt_to_send).replace("\n", "").replace("Answer:","").lower()
    return result_from_chatgpt
## Web Scrapping

url_input = "https://news.yahoo.com"
# url_input = "https://laion.ai/blog/" # OK
# url_input = "https://www.euronews.com/tag/artificial-intelligence" # NOK
# url_input = "https://www.theguardian.com/international" #OK
# url_input = "https://www.bloomberg.com/europe" #NOK
# url_input = "https://news.google.com/home?hl=en-US&gl=US&ceid=US:en" # OK

### USER INPUT HERE
if validators.url(url_input):
    url_to_watch = st.text_input("Input your URL here", url_input)
    ## Process website and save content to file
    text_from_webpage = process_website(url_to_watch, "output.txt")
    text_from_webpage = text_from_webpage[:HARD_LIMIT_CHAR]
else: 
    print("URL not valid")
    ### UI OUTPUT HERE
    #st.write("URL not valid")  

prompt_news = "Below is an html version of a news website. It contains news articles. Find the titles of news articles on this website. Do not make up article titles. List all the article titles and their metadata if it exists like date or author. Limit yourself to the first 5. In JSON format, using these keys \"title\", \"metadata\". No Other text."

result_from_chatgpt = prompt_to_llm_response(text_from_webpage,prompt_news)

result_from_chatgpt_processed = result_from_chatgpt.encode('ascii', 'ignore')

print(json.dumps(json.loads(result_from_chatgpt_processed), indent=4))

file_path = "gpt_out.txt"

parsed_articles = json.loads(result_from_chatgpt)
#Logging
file_path = "output_gpt.txt"
with open(file_path, "w") as file:
    file.write(result_from_chatgpt)
print("Variable content saved to the file:", file_path)


#with open('final_output.json', 'w') as f:
#  print("The json file is created")

### USER INPUT HERE
#topic_of_interest = "Should AI be open sourced?"
topic_of_interest = "Ukraine War"

empty_list = []
i = 0

for item in json.loads(result_from_chatgpt_processed):
    i+=1
    output_filename = "article_text"+str(i)+".txt"

    article_title = item['title']
    article_link = get_link_based_on_article_name_via_google(article_title, url_to_watch)
    
    new_item = {
        'title': item['title'],
        'metadata': item['metadata'],
        'link': article_link,
    }

    relation_exists = prompt_similarity_to_llm_response(article_title,topic_of_interest)
        
    if relation_exists == "strongly agree" or relation_exists ==  "agree" :
        article_text = process_website(article_link, output_filename)

        # Summarize article
        prompt_article = "Summarize the following text in 3 sentences: "
        article_summary = prompt_to_llm_response(article_text,prompt_article)

        # Answer the question
        prompt_content = "If user input is a question provide an answer, otherwise summarise content relevant to the input topic. Answer in one sentence".format(topic_of_interest)
        user_question_answer = prompt_to_llm_response(article_text,prompt_content)
    
        new_item["summary"]=article_summary
        new_item["answer"]=user_question_answer
        new_item["related?"]=relation_exists
        
    #else: print("not relevant")
        
    empty_list.append(new_item)

output_json = json.dumps(empty_list, indent=4)

### UI OUTPUT HERE
with open("output.json", "w") as outfile:
    outfile.write(output_json)
