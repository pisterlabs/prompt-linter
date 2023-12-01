import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

# prompt = "Who won the 2020 Summer Olympics men's high jump?"

# x = openai.Completion.create(
#     prompt=prompt,
#     temperature=0,
#     max_tokens=300,
#     model=COMPLETIONS_MODEL
# )["choices"][0]["text"].strip(" \n")

# print(x)

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }



df = pd.read_csv('docs.csv')
df = df.set_index(["title", "heading"])




embeddings = compute_doc_embeddings(df)

file = open('embeddings.binary', 'wb')
pickle.dump(embeddings, file)
file.close()


