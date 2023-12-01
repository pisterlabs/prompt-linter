import openai
import os
import numpy as np
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

df = pd.read_csv("embedded_fortune_1k_revenue.csv")
# print(df.head())

query = "what was Amazon revenues"

query_vector = get_embedding(text=query)
# print(query_vector)
print(cosine_similarity(np.array(np.matrix(df['ada_embedding'][0])).ravel(), query_vector))
df['cosine_similarity'] = df['ada_embedding'].apply(
                            lambda v: cosine_similarity(np.array(np.matrix(v)).ravel(), query_vector))

print(df['cosine_similarity'])
most_similar_index = np.argmax(df['cosine_similarity'])
print(f"Most similar sentence:: {df['info'][most_similar_index]}")

embeded_prompt = f"""answer only if you are 100% certain
Reference: {df['info'][most_similar_index]}

Question: {query}
Answer: """

print("------------------------------")
print(embeded_prompt)
print("------------------------------")

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=embeded_prompt,
    max_tokens=300,
)
print(f"openai answer: {response['choices'][0]['text']}")

"""
Most similar sentence:: Amazon has $469,822 revenues, $1,658,807.30 market value and 1,608,000 employees
------------------------------
answer only if you are 100% certain
Reference: Amazon has $469,822 revenues, $1,658,807.30 market value and 1,608,000 employees

Question: what was Amazon revenues
Answer:
------------------------------
 $469,822
"""