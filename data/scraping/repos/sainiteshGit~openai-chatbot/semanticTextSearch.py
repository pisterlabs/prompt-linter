import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai

openai.api_key = 'YOUR_OPEN_API_KEY'

datafile_path = "/Users/SaiNitesh/Projects/openai-chatbot/content/data/search/fine_food_reviews_with_embeddings_500.csv"

df = pd.read_csv(datafile_path)

df["embedding"] = df.embedding.apply(eval).apply(np.array)

##### search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

query=input('What do you want to search in reviews first time? \n')
results = search_reviews(df, query, n=2)
print('################################################################################################################### RESULTS #######################################################################')
print(results);
print('###########################################################################################################################################################')
query=input('What do you want to search in reviews second time? \n')
print('#####################################################################################################################################################')
