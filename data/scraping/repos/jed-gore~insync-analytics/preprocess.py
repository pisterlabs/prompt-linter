import json
import pandas as pd

# import nltk
# import numpy as np
# from more_itertools import split_after

# from nltk import word_tokenize, pos_tag, ne_chunk
# from nltk import RegexpParser
# from nltk import Tree
from textblob import TextBlob


# import pandas_gpt
import openai

with open("data_sources.json") as f:
    data = f.read()

js = json.loads(data)
key = js["openai"]["api_key"]


replacers = {
    "Q/Q": "quarter over quarter growth in ",
    "Y/Y": "year over year growth in ",
    "(Y/Y)": "year over year growth in",
    "TTM": "trailing twelve months",
    "/": " per ",
    "Incr.": "incremental",
    "$": "dollars of ",
    "IS": "income statement",
    "BS": "balance sheet",
    "KM": "key metrics",
}


df = pd.read_csv("DATA/AMZN/AMZN_actuals.csv", encoding="ISO-8859-1")

df["LineItem"] = (
    df.LineItem.str.replace("[...…]", "")
    .str.split()
    .apply(lambda x: " ".join([replacers.get(e, e) for e in x]))
)
df["Category"] = (
    df.Category.str.replace("[...…]", "")
    .str.split()
    .apply(lambda x: " ".join([replacers.get(e, e) for e in x]))
)


def get_noun_phrases(text):
    blob = TextBlob(text).noun_phrases
    return ",".join(blob)


df["key_phrases"] = df["LineItem"].apply(lambda sent: get_noun_phrases((sent)))

if key == "":
    print(f"get an api key at https://platform.openai.com/account/api-keys")
else:
    openai.api_key = key

print(df.head())
