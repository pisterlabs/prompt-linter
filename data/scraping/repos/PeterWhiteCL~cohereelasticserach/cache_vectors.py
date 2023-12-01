import json

import pandas as pd
from config import COHERE_MODEL, DATA_PATH, EMBED_COLUMN, COHERE_API_KEY
from utils import get_cohere_embedding

df = pd.read_csv(DATA_PATH).fillna("").reset_index(drop=True)


# create list of texts
texts = list(df['abstract'].str.strip())

# embed all texts with cohere client
embed_list = get_cohere_embedding(texts, model_name=COHERE_MODEL, input_type="search_document")

# create a lookup table of text:vector
cache = dict(zip(texts, embed_list))

# dump out cache
with open("cache.jsonl", "w") as fp:
    json.dump(cache, fp)
