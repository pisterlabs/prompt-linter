from dotenv import dotenv_values
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import os

config = dotenv_values()
# openai.organization = config['YOUR_ORG_ID']
try:
    openai.api_key = config['OPENAI_API_KEY']
except:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
# openai.Model.list()


def toia_answer(query, data, k=1):
    print("Query", query)
    embedding = get_embedding(query, engine='text-search-ada-query-001')
    print("B")
    data['similarities'] = data.ada_search.apply(lambda x: cosine_similarity(x, embedding))
    print("C")
    res = data.sort_values('similarities', ascending=False).head(k)
    print("D")

    ada_similarity_score = res.similarities.values[0]
    print("Similarity Score", ada_similarity_score)
    if ada_similarity_score > 0.29:
        return res['answer'].values[0], res['id_video'].values[0], res['language'].values[0], ada_similarity_score
    else:
        df_noanswers = data[data['type'] == "no-answer"]
        if df_noanswers.shape[0] > 0:
            answers = df_noanswers.sample(n=1)
            return answers['answer'].values[0], answers['id_video'].values[0], answers['language'].values[0], ada_similarity_score
        else:
            return "You haven't recorded no-answers", "204", "No Content"