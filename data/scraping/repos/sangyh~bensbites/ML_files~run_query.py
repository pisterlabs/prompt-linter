
from ML_files.utils import utils
import pandas as pd
import os
import openai
from ast import literal_eval
from django.core.cache import cache
import json

def load_file():
    print('Loading file')
    ### read embeddings
    df = pd.read_csv('ML_files/embedded_bensbites.csv')
    #print(df.columns)
    #df.ada_embedding = df.ada_embedding.apply(literal_eval)
    df.item_url = df.item_url.apply(literal_eval)
    return df

def load_topics():
    print('Loading topics json')
    
    ### read existing json
    with open('ML_files/topicid2labels.json', 'r') as f:
        # Load the JSON data into a Python object
        topics2labels = json.load(f)

    return topics2labels

def get_topics():
    topics2labels = cache.get('all_topics')
    if topics2labels is None:
        topics2labels = load_topics()
        cache.set('all_topics', topics2labels, timeout=3600)

    print('Topics loaded')

    all_topics = topics2labels.values()
    return all_topics

def filter_topics(topic):
    print('Topic to filter:', topic)
    df = cache.get('all_embeddings')
    topics2labels = cache.get('all_topics')

    if df is None:
        df = load_file()
        cache.set('all_embeddings', df, timeout=3600)

    if topics2labels is None:
        topics2labels = load_topics()
        cache.set('all_topics', topics2labels, timeout=3600)    

    results = utils.filter_items(df, topic)

    return results

def get_search_results(query):
    ROOT_DIR = os.path.abspath('./')

    if (os.environ.get("OPENAI_API_KEY") == None):
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT_DIR, 'ML_files', 'config', 'conf', '.env'))

    openai.api_key = os.getenv("OPENAI_API_KEY")

    df = cache.get('all_embeddings')
    
    if df is None:
        df = load_file()
        cache.set('all_embeddings', df, timeout=3600)
    print('Embeddings loaded')


    if query=='':
        results = df[['url', 'section','item_text','item_url']].head(10).to_json(orient = "records")
        
    else:
        # Process the input and print the results
        results = utils.search_items(df, query, n=5)
    return results
