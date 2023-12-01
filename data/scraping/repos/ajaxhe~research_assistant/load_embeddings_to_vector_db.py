import ast
import configparser
import os
import openai
import pandas as pd
import pinecone
import time

from datetime import date, datetime
from openai.embeddings_utils import distances_from_embeddings
from tqdm.auto import tqdm

config = configparser.ConfigParser()
config.read('default.cfg')
d_conf = config['DEFAULT']

# Init openai param
openai.api_type = d_conf['api_type']
openai.api_base = d_conf['api_base'] 
openai.api_version = d_conf['api_version']

deployment_id = d_conf['fast_llm_model_deployment_id']
temperature = int(d_conf['temperature'])

# Define index name
index_name = 'research-assistant'

max_tokens = int(d_conf['max_tokens'])
max_len=6000

date_format_str= '%Y-%m-%d'

def create_context(question, df, max_len=max_len):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
            input=question, 
            deployment_id=d_conf['embedding_model_depleyment_id'],
            #engine='text-embedding-ada-002',
            )['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def load_to_vector_db(df):
    '''
    metadata_config = {
        'indexed': ['p_date', 'p_date_ut', 'domain', 'f_domain']
    }
    '''
    # Check if index already exists, create it if it doesn't
    if index_name not in pinecone.list_indexes():
        #pinecone.create_index(index_name, dimension=1536, metadata_config=metadata_config)
        pinecone.create_index(index_name, dimension=1536)

    # Connect to the index and view index stats
    index = pinecone.Index(index_name)
    print(index.describe_index_stats())

    batch_size = 10  # how many embeddings we create and insert at once
    
    # Convert the DataFrame to a list of dictionaries
    chunks = df.to_dict(orient='records')
    
    # Upsert embeddings into Pinecone in batches of 100
    for i in tqdm(range(0, len(chunks), batch_size)):
        i_end = min(len(chunks), i+batch_size)
        meta_batch = chunks[i:i_end]
        ids_batch = [str(x['id']) for x in meta_batch]
        embeds = [x['embeddings'] for x in meta_batch]
        meta_batch = [{
            'domain': x['domain'],
            'f_domain': x['f_domain'],
            'url': x['url'],
            'p_date': x['p_date'],
            'p_date_ut': int(time.mktime(datetime.strptime(x['p_date'], date_format_str).timetuple())),
            'n_tokens': x['n_tokens'],
            'text': x['text'],
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        index.upsert(vectors=to_upsert)

    print(f"total upsert vector: {i_end}")

def main():
    df = pd.read_csv(d_conf['csv_file'])
    #print(df.head())

    # convert str to list
    df['embeddings'] = df.embeddings.apply(lambda x: ast.literal_eval(x))

    # Initialize connection to Pinecone
    pinecone.init(api_key=d_conf['pinecone_api_key'], environment=d_conf['pinecone_api_env'])

    load_to_vector_db(df)

if __name__ == '__main__':
    main()

