import os, time, sys
import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding
from pathlib import Path
import yaml
import logging
config = yaml.safe_load(open("config.yml"))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


embedding_model = config['ada']['engine'] # using text-search-ada-doc-001 instead of text-embedding-ada-002 becos of ES 1024 dims limit temporarily
embedding_encoding = config['ada']['encoding']  #r50k_base for text-search-ada-doc-001. cl100k_base is the encoding for text-embedding-ada-002
max_tokens = 1000  # the maximum for text-embedding-ada-002 is 8191

# limits to reduce credits usage
n_samples = config['ada']['n_samples']
raw_data_path = os.path.join(config['data']['data_folder'], config['data']['raw_data'])
ada_emb_path = os.path.join(config['data']['data_folder'], config['data']['data_w_ada_emb'])

def encode():
    print("Encoding data")

    # Query some data to test
    try:
        data_df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print("FileNotFoundError - Please query data and try again")
        return

    # #omit reviews that are too long to embed
    encoding = tiktoken.get_encoding(embedding_encoding)
    data_df["n_tokens"] = data_df.combined.apply(lambda x: len(encoding.encode(x)))
    data_df = data_df[data_df["n_tokens"] <= max_tokens]
    print(data_df.shape)

    # Get Embeddings
    n_tokens_embedded = 0
    count = 1

    # # omit rows which has been embedded b4
    try:
        embedded_df = pd.read_csv(ada_emb_path)
        embedded_df.dropna(subset=['embedding'], inplace=True)
        print("embedded_df.shape: {}".format(str(embedded_df.shape)))
        data_df = data_df[~data_df['id'].isin(embedded_df['id'])]
        count = len(data_df)
    except FileNotFoundError:
        embedded_df = pd.DataFrame(columns=["id", "title", "desc", "combined", "embedding"])
        pass

    to_sample = n_samples - len(embedded_df)
    to_embed_df = data_df.sample(to_sample, random_state=999)
    to_embed_df['embedding'] = None
    to_embed_df['embedding'] = to_embed_df['embedding'].astype('object')
    cols = ["id", "title", "desc", "combined", "embedding"]
    print("to_embed_df.shape: {}".format(str(to_embed_df.shape)))

    count = 0
    step = 100
    for index, row in to_embed_df.iterrows():
        id, text = row['id'], row['combined']
        query_vector = get_embedding(text, engine=embedding_model)
        to_embed_df.at[index, 'embedding'] = query_vector
        count +=1
        # vectors_list.append(query_vector)
        if count % step == 0:
            new_embedded_df = pd.concat([embedded_df[cols], to_embed_df[cols]])
            new_embedded_df.to_csv(os.path.join(config['data']['data_folder'], config['data']['data_w_ada_emb']), index=False)
            query_ids, queries, vectors_list = [], [], []
            logging.info("Encoded {} queries".format(len(to_embed_df['embedding'].dropna())))
            time.sleep(3)

encode()
# index_data()






