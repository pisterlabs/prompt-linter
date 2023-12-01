from txtai.embeddings import Embeddings
import json
import pandas as pd
import numpy as np
import os

import sqlite3
from dotenv import load_dotenv

"""
This script is used to do word embedding using OpenAI API.
"""
def main():
    df = pd.read_csv('./output/FullDF.csv')
    df.head()
    df['embedding'] = df['Name'] + ', ' + df['Description'] + ', ' + df['Type']

    import openai
    from openai import OpenAI

    # api_key = None
    api_key=os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=api_key)

    def get_embedding(text, model="text-embedding-ada-002"):
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    df = df.dropna().reset_index(drop=True)

    df['ada_embedding'] = df.embedding.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    df.to_csv('./output/ada_embeddings.csv', index=False)
main()