
from typing import List, Tuple
from multiprocessing import Pool
import re
import cohere
import time 
import os
import functools
import cProfile
import pathlib as pl
import pandas as pd
import ast
import numpy as np
import re

parent = pl.Path(__file__).parent  

def tweet_to_dict(tweet: tuple)-> dict:
    Dict = {
            "Author": tweet[1][1], 
            "Date": tweet[1][4], 
            "Content": tweet[1][0],
            "Likes": tweet[1][2],
            "Retweets": tweet[1][3]
        }
    return Dict


def process_dataframe(df):
    df['content_embedding'] = df['content_embedding'].apply(lambda x: ast.literal_eval(x))
    df['date'] = df['date'].apply(lambda x: get_date(x))  
    df['content'] = df['content'].apply(lambda x: str(x))
    return df

def similarity_search_(n, embedding, indexer):
    # Perform a similarity search
    results = indexer.search(embedding, n)
    # Print the results
    for result in results:
        print(result)

def find_hashtags(text) -> list:
    if not isinstance(text, str):
        text = f'{text}'
    pattern = re.compile(r"#(\w+)")
    hashtags = pattern.findall(text)
    return hashtags

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        path = f"{parent}/profiling/"
        if not os.path.exists(path):
            os.makedirs(path)
        profiler.dump_stats(path + f"{func.__name__}.prof")
        return result

    return wrapper

def token_count(text: str) -> int:
    return len(text.split())*3    

def list_to_string(inp: List[Tuple[str,tuple]]) -> str:
    return "\n".join([f"{pred}{elm}" for pred,elm in inp]) if len(inp) > 0 else " "
    
def convert_to_BLOB(embedding):
    out = np.array(embedding) # np array to bytes for blob data in sqlite, float 64 is the default
    return out.tobytes()

def embed(text: List[str]):
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    time.sleep((100/60) + 0.01)
    response = co.embed(
    texts=text,
    model='small',
    )
    return response

def create_embedding_bytes(text: str) -> bytes:
    if type(text) is list:
        raise Exception("text is a list of strings, please pass a string")
    
    response = embed([text])    
    out_lst = convert_to_BLOB(response.embeddings[0])
    return out_lst

def create_embedding_nparray(text: str) -> np.array:
    if type(text) is list:
        raise Exception("text is a list of strings, please pass a string")
    response = embed([text])    
    out_lst = [np.array(embedding) for embedding in response.embeddings]
    return out_lst

# After retrieving from SQLite

def convert_bytes_to_nparray(embedding_bytes:bytes) -> np.array:
    '''Converts a byte stream to a numpy array'''
    try: 
        embedding_np = np.frombuffer(embedding_bytes, dtype=np.float64)  
    except TypeError as e:
        print(e, "build db from scratch again")

    return embedding_np

def get_date(data):
    pattern = r'\d{4}-\d{2}-\d{2}'
    if data is not None:
        match = re.search(pattern, data)
        if match:
            return match.group(0)
    else: 
        return None

