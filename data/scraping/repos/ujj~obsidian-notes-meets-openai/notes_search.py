import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity

openai.api_key = "YOUR_OPEN_AI_API_KEY"


search_text = "Social identity theory says that our social behaviour is determined by our character and motivations as well as by our memberships in groups. The most concerning part of belonging to a group is the in-group bias that lead to us v them problems (engineering v product, sales v marketing, etc"


def load_embeds(filename):
    df = pd.read_csv(filename)
    df['babbage-search'] = df.babbage_search.apply(eval).apply(np.array)
    return df


def find_matching_notes(daf, text, n=5):
    embedding = get_embedding(text, engine='text-search-babbage-query-001')
    daf['similarities'] = daf['babbage-search'].apply(
        lambda x: cosine_similarity(x, embedding))

    res = daf.sort_values('similarities', ascending=False).head(n). \
        combined.str.replace('Title: ', '').str.replace('; Content:', ': ')

    for r in res:
        print("------\n")
        print(r[:200])
        print("--------\n")
    return res


if __name__ == '__main__':
    dtf = load_embeds("ujjwal_notes_embeddings.csv")
    find_matching_notes(dtf, search_text)
