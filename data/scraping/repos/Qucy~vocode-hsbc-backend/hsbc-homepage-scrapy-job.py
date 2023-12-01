""" This is a DAG to scrapy HSBC HK homepage info and deploy on GCP MapleQuad's Airflow
The knowledge will be stored in a txt file and upload to GCS bucket
"""
import re
import os
import json
import requests
import openai
import psycopg2
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import Union, List

# init variables
homepage_url = os.getenv('hsbc_homepage_url')
wealth_insigths_articles = os.getenv('wealth_insigths_articles')
chinese_pattern = re.compile("[\u4e00-\u9fff\u3400-\u4dbf]+")

# pgsql configuration
host = os.getenv('pg_host')
dbname = os.getenv('pg_db_name')
user = os.getenv('pg_user')
password = os.getenv('pg_password')
sslmode = os.getenv('pg_sslmode')

# openai configuration
openai.api_key = os.getenv('openai_api_key')
openai.api_version = os.getenv('openai_api_version')
openai.api_type = os.getenv('openai_api_type')
openai.api_base = os.getenv('openai_api_base')

# Construct connection string
conn_string = f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
conn = psycopg2.connect(conn_string) 

# ================================== OPENAI Response model ==================================
class OPENAICompletionResponseChoiceMessage(BaseModel):
    """
    OPENAI response model
    """
    role: str
    content: str

class OPENAICompletionResponseChoice(BaseModel):
    """
    OPENAI response model
    """
    index: int
    finish_reason: Union[str, None] = None
    message: Union[OPENAICompletionResponseChoiceMessage, None] = None

class OPENAICompletionResponseUsage(BaseModel):
    """
    OPENAI response model
    """
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class OPENAICompletionResponse(BaseModel):
    """
    OPENAI response model
    """
    id: str
    object: str
    created: int
    model: str
    choices: List[OPENAICompletionResponseChoice] = list()
    usage: Union[OPENAICompletionResponseUsage, None] = None
# ================================== OPENAI Response model ==================================

def scrape_content_by_url(url: str):
    """ scrape content by url
    """
    # Send a GET request to the webpage URL
    response = requests.get(url, timeout=5)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the text content and remove excess whitespace
    text = soup.get_text()
    text = re.sub('\s+', ' ', text).strip().lower()
    text = chinese_pattern.sub("", text)
    
    return text


def retreive_urls_by_parent_url():
    """ retreive urls by parent url
    """
    # Send a GET request to the webpage URL
    response = requests.get(homepage_url, timeout=5)
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find the url start with a
    a_tags = soup.find_all('a')
    # retrieve all the link
    urls = [a_tag.get('href') for a_tag in a_tags if a_tag is not None and a_tag.get('href') is not None]
    # remove all the link start with http or https
    urls = [url for url in urls if not url.startswith(('http', '#'))]
    # remove other language url
    urls = [url for url in urls if url not in ('/', '/zh-hk/', '/zh-cn/')]
    return urls

def knowledge_extraction(key_words:str, content: str):
    # genrate prompt
    system_prompt = {"role" : "system", "content" : f"You are a knowledge extract assistant, help people extract key information."}
    user_prompt = {"role" : "user", "content" : f"Extract <{key_words}> related information from <{content}> in english around 1000 words."}
    # generate messages
    messages = [system_prompt, user_prompt]
    # generate response
    response = openai.ChatCompletion.create(
        engine=os.getenv('openai_engine'),
        max_tokens=1000,
        temperature=.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        messages=messages,
    )
    # conver to data model
    openAICompletionResponse = OPENAICompletionResponse.parse_obj(response)
    # return content
    if len(openAICompletionResponse.choices) == 0:
        return ""
    else:
        return openAICompletionResponse.choices[0].message.content
    
def embedding_calculation(content: str):
    """Embedding calculation
    """
    response = openai.Embedding.create(
        input="content",
        engine="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

def save_to_pgsql(url, keywords, content, embedding):
    """ save to pgsql
    """
    try:
        # remove special characters from content to avoid insert error
        content = content.replace("'", "''")
        # generate sql
        sql = f"""
        delete from hsbc_homepage_content where url = '{url}';
        insert into hsbc_homepage_content (url, keywords, content, embedding) values ('{url}', '{keywords}', '{content}', '{embedding}');
        """
        # create a cursor
        cursor = conn.cursor()
        # execute sql
        cursor.execute(sql)
        # commit the changes to the database
        conn.commit()
    finally:
        # close communication with the database
        cursor.close()


@task(task_id='scrapy')
def extract_knowledge():
    """
    Extract knowledge from hsbc homepage and wealth insights
    """
    # extract home page child urls
    urls = retreive_urls_by_parent_url()
    # loop and extract content
    for url in urls:
        try:
            # retreive content
            content = scrape_content_by_url(homepage_url + url)
            key_words = url.replace('/', ' ')
            # skip the content if less than 50 words
            if len(content) < 50:
                print('Skip current url -> content less than 50 words with url=%s' % url)
                continue
            # send content to LLM to do summaraization before feed into vector store
            knowledge = knowledge_extraction(key_words, content)
            # send summary to LLM to calc embedding
            embedding = embedding_calculation(knowledge)
            # save into pg vector store
            save_to_pgsql(url, key_words, knowledge, embedding)
            # log
            print(f"Successfuly saved embedding for {url}, content length {len(content)}, knowledge length {len(knowledge)}, embedding length {len(embedding)}")
           
        except Exception as e:
            print(f'Skip current url -> error occurred when scraping content from [{url}] with error:[{e}]')

    # extract wealth insights content
    try:
        # extract wealth insights content
        response = requests.get(wealth_insigths_articles, timeout=10)
        if response.status_code == 200:
            data = json.loads(response.text)
            for article in data:
                title, href = article["title"], article["href"]
                # retreive content
                content = scrape_content_by_url(href)
                # send content to LLM to do summaraization before feed into vector store
                knowledge = knowledge_extraction(key_words, content)
                # send summary to LLM to calc embedding
                embedding = embedding_calculation(knowledge)
                # save into pg vector store
                save_to_pgsql(href, title, knowledge, embedding)
                # log
                print(f"Successfuly saved embedding for {href}, content length {len(content)}, knowledge length {len(knowledge)}, embedding length {len(embedding)}") 
        else:
            print("Error: Could not retrieve JSON data")
    except Exception as e:
        print('Error occurred when scraping wealth insights content with error=%s' % e)


with DAG(
    'hsbc-knowledge-scrapy-job',
    default_args={
        'depends_on_past': False,
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 0,
        'dagrun_timeout': timedelta(hours=4) 
    },
    description='DAG to scrapy HSBC HK homepage knowledge and weath insights and upload to GCS bucket',
    schedule_interval='0 2 * * *',
    start_date=datetime(2023, 6, 29),
    catchup=False,
    tags=['hsbc','homepage','scrape'],
) as dag:

    t1 = extract_knowledge()

    t1

# if __name__ == "__main__":
#     extract_knowledge()

