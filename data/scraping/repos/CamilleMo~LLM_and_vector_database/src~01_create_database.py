import openai
import pandas as pd

from .load_config import load

df = pd.read_excel("data/documents.xlsx")


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


openai.api_key = load("config.yaml")["tokens"]["openai"]

df["embeddings"] = df.apply(lambda x: get_embedding(x["document"]), axis=1)
# save df to a csv file
df.to_csv("data/documents_processed.csv", index=False)
