# summerise the larage text if it is more than 4000 OpenAI tokens
import pandas as pd
import numpy as np
import openai
import os
import redisembeddings
from translator import translate
from openai.embeddings_utils import get_embedding
from tenacity import retry, wait_random_exponential, stop_after_attempt
from redisembeddings import set_document
from formrecognizer import analyze_read
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Maxi LangChain token limitation for Davinci model
DEFAULT_CHUNK_LIMIT = 20000
CHUNK_LIMIT = 8000
def initialize(engine='davinci'):
    openai.api_type = "azure"
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_version = "2022-12-01"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    CHUNK_LIMIT = os.getenv("OPENAI_CHUNK_LIMIT")
    redisembeddings.initialize()
    

# summarise the file content and embed
def convert_file_and_add_summarisation_and_embed(fullpath, filename, enable_translation=False):
    
    # Extract the text from the file
    text = analyze_read(fullpath)
    if enable_translation:
        text = list(map(lambda x: translate(x), text))
    summary_text = "".join(text)
    return add_summarisation_embeddings(summary_text,filename, os.getenv('OPENAI_SUMMARISATION_ENGINE_DOC', 'text-davinci-003'),os.getenv('OPENAI_EMBEDDINGS_ENGINE_DOC', 'text-embedding-ada-002'))
    

def add_summarisation_embeddings(text, filename, summarise_engine="text-davinci-003",embed_engine="text-embedding-ada-002"):
    summarisation = chunk_and_summarise_embed(text, filename, summarise_engine,embed_engine,CHUNK_LIMIT)
    
    if summarisation:
        # Store embeddings in Redis
        set_document(summarisation)
        return True
    else:
        print("No summarisation and embeddings were created for this document as document is invaild or unable to read. Please check the document")
        return False


def chunk_and_summarise_embed(text: str, filename="", summarise_engine="text-davinci-003",embed_engine="text-embedding-ada-002", chunk_limit=8000):
    # set the maximum chunks limit to 8000
    CHUNK_LIMIT = chunk_limit
    if CHUNK_LIMIT > DEFAULT_CHUNK_LIMIT:
            CHUNK_LIMIT = DEFAULT_CHUNK_LIMIT

    full_data = {
        "text": text,
        "filename": filename,
        "search_embeddings": None
    }
    
    # call LangChain TokenTextSplitter to split the text semantically 
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size
        chunk_size = CHUNK_LIMIT,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=0,
        #chunk_overlap  = 200,
        #length_function = len,
        )
    split_texts = text_splitter.create_documents([text])

    # get summarisation for each token
    summary = ''
    for x in range(len(split_texts)):
        # get summarisation for each chunks and append
        # send the pervious summmerisation text
        response = get_summarise((str(split_texts[x])), summarise_engine)
        summary += f"{response['choices'][0]['text']}\n"               
        
    # get the embeddings for summarisation
    full_data['text'] = summary
    full_data['search_embeddings'] = get_embedding(summary, embed_engine)

    return full_data

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_summarise(text: str, model="text-davinci-003") -> list[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    # add end of line Tl;dr for summarisation
    text += "\n"
    text += "Tl;dr"
    
    # call the summarisation before embeddings
    return openai.Completion.create(engine=model,prompt=text, max_tokens=100,temperature=0.0,top_p=1,frequency_penalty=0,presence_penalty=0,stop=None)    
