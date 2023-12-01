#!/usr/bin/env python
# coding: utf-8

# In[8]:


#if needed 
#!set_db.sh


# In[24]:


import re
import numpy as np
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import os, json
from nltk.tokenize import sent_tokenize,word_tokenize
import datetime
import os
import openai
import time
import configparser
import hashlib

openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_TOKENS = 100
EMBEDDING_SIZE = 1536
special_splitter="#!#" #used to always split text into chunks on that token.



#cross project import
from get_full_text import clean_text,retrieve_full_text 



def log(text, end="\n"):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {text}", end=end)
import re
from typing import List

def split_sentence(sentence: str, max_length: int=MAX_TOKENS) -> List[str]:
    comma_parts = sentence.split(', ')
    chunks = []
    current_chunk = []

    for part in comma_parts:
        words = part.split()

        if len(current_chunk) + len(words) + 1 <= max_length:  # +1 for space between comma parts
            current_chunk.extend(words)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

            if len(words) > max_length:
                for i in range(0, len(words), max_length):
                    sub_chunk = words[i:i + max_length]
                    chunks.append(" ".join(sub_chunk))
            else:
                current_chunk.extend(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_splitter(input_string: str, 
                   max_length: int = MAX_TOKENS, 
                   special_string: str = special_splitter,
                   prefix:str = None,
                   clean:bool = True) -> List[str]:

    if clean:
        input_string = clean_text(input_string)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', input_string)
    flat_sentences = []
    for sentence in sentences:
        flat_sentences.extend(sentence.split(special_string))
    sentences = flat_sentences

    #deal with sentences that are longer than max_length, by splitting them into max chunks of ","
    sentences = [split_sentence(sentence, max_length) for sentence in sentences]
    sentences = [item for sublist in sentences for item in sublist]  # Flatten the list of sentences

    chunks = []
    chunk = []

    for sentence in sentences:
        words = sentence.split()
        
        if len(chunk) + len(words) + 1 <= max_length:  # +1 for space between sentences
            chunk.extend(words)
        else:
            #adding words to the chunk would exceed the max length
            if chunk:
                chunks.append(" ".join(chunk))
                chunk = []    
            chunk.extend(words)
    #last chunk
    if chunk:
        chunks.append(" ".join(chunk))

    if prefix is not None:
        chunks = [prefix + chunk for chunk in chunks]

    return chunks

# Add these lines to register np.ndarray for psycopg2
def adapt_np_array(array):
    return AsIs(np.array(array).tolist())
register_adapter(np.ndarray, adapt_np_array)

def fetch_openai(model,chunk):
    response = openai.Embedding.create(
        input=chunk,
        model=model
    )
    return response

def openai_embeddings(model,chunks):
    if isinstance(chunks, str):
        chunks = [chunks]
    i=0
    sentence_embeddings =[]
    for chunk in chunks:
        i+=1
        if i % 50 == 0:
            log(f"Doing -> Embedding chunk: {i} of {len(chunks)}. Chunk ({len(chunk.split())} tokens):\n\t {chunk}...")
        
        try:
            response = fetch_openai(model,chunk)
        except Exception as e:
            log(f"Error -> Retry in 5 seconds.")
            time.sleep(5)
            response = fetch_openai(model,chunk)
        sentence_embeddings.append(response['data'][0]['embedding'])
    return sentence_embeddings

def entry_id_exists(entry_id, conn):
    c = conn.cursor()
    c.execute("SELECT 1 FROM embeddings_openai WHERE entry_id = %s LIMIT 1", (entry_id,))
    return c.fetchone() is not None


# Function to process a project
def process_project(thread_id, project):
    retrieve_full_text(project)
    project["keywords"] = project["keywords"].replace(";", ". ").replace(",", ". ")

    local_counter = 1
    with psycopg2.connect(**db_config) as conn:
        entry_id = "P:"+",".join(project['ids'])
        if entry_id_exists(entry_id, conn):
            log(f"Skipping -> Project {project['title']} already exists in the table.")
            return
        
        retrieve_full_text(project)
        chunks = chunk_splitter(project["full_text"], prefix="TX: ")
        chunks.extend(chunk_splitter(project["title"]   , prefix="Title: "))
        chunks.extend(chunk_splitter(project["abstract"], prefix="Abstract: "))
        chunks.extend(chunk_splitter(project["keywords"], prefix="Keywords: "))

        log(f"Starting -> {len(chunks)} chunks for project {project['title']}.")
        sentence_embeddings = openai_embeddings(model,chunks)
        c = conn.cursor()
        for chunk, embedding in zip(chunks, sentence_embeddings):
            unique_id = thread_id * 1000000 + local_counter
            chunk = chunk.replace('\x00', ' ')  # Replace NUL characters with a space
            c.execute("INSERT INTO embeddings_openai (id, entry_id, chunk, embedding) VALUES (%s, %s, %s, %s::VECTOR)", (unique_id, project_id, chunk, embedding))
            local_counter += 1
        conn.commit()
        log(f"Done -> Project {project['title']}")

def process_use_case(use_case,id):
    local_counter = 0
    with psycopg2.connect(**db_config) as conn:
        entry_id = "UC:"+use_case[:20]+"#"+hashlib.md5(use_case.encode()).hexdigest()
        if entry_id_exists(entry_id, conn):
            log(f"Skipping -> Use Case {entry_id} already exists in the table.")
            return

        chunks = chunk_splitter(use_case)

        log(f"Starting -> {len(chunks)} chunks for project {entry_id}.")
        sentence_embeddings = openai_embeddings(model,chunks)
        c = conn.cursor()
        for chunk, embedding in zip(chunks, sentence_embeddings):
            unique_id = id*1000000 + local_counter
            chunk = chunk.replace('\x00', ' ')  # Replace NUL characters with a space
            c.execute("INSERT INTO embeddings_openai (id, entry_id, chunk, embedding) VALUES (%s, %s, %s, %s::VECTOR)", (unique_id, entry_id, chunk, embedding))
            local_counter += 1
        conn.commit()
        log(f"Done -> Project {entry_id}")




# In[25]:


if __name__ == "__main__":
    reset_db = True #drop table and create new one

    # Database configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    db_config = {
        'database':   config['DB']['database'],
        'user':     config['DB']['username'],
        'password': config['DB']['password'],
        'host':     config['DB']['host'],
        'port':     int(config['DB']['port'])
    }  

    
    model = config['OPENAI']['model']



    # Create a folder to store text files
    text_folder = "text_files"

    with psycopg2.connect(**db_config) as conn:
        c = conn.cursor()
        if reset_db:
            log("Resetting database.")
            c.execute("DROP TABLE IF EXISTS embeddings_openai;")
            c.execute("DROP SEQUENCE IF EXISTS embeddings_openai_id_seq;")
        c.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        c.execute('CREATE SEQUENCE IF NOT EXISTS embeddings_openai_id_seq;')
        c.execute(f'CREATE TABLE IF NOT EXISTS embeddings_openai (id INTEGER PRIMARY KEY DEFAULT nextval(\'embeddings_openai_id_seq\'), entry_id TEXT, chunk TEXT, embedding VECTOR({EMBEDDING_SIZE}));')
        
        conn.commit()


    # ingest projects
    # projects_file = "data/digital_agriculture_projects.json"
    # with open(projects_file, "r") as f:
    #     projects = json.load(f)
    #    #Process texts and save embeddings into the database using 8 threads
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     for i, _ in enumerate(executor.map(process_project, range(len(projects[:])), projects[:])):
    #         pass 
    
    # ingest use cases
    use_cases_file = "data/digital_agriculture_use_cases.json"
    with open(use_cases_file, "r") as f:
        use_cases = json.load(f)

    for id,use_case in enumerate(use_cases,1):
        process_use_case(use_case,id)

    # ingest datasets

    # Initialize counter and lock
    counter = 1
    counter_lock = Lock()




# In[ ]:




