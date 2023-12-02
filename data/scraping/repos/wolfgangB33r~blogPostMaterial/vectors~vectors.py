import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)

text_a = "i had troubles downloading the app"

text_b = "ich hatte probleme die app herunterzuladen"

text_c = "the book 'Lord of the Rings' is a great read"

model_id = "text-embedding-ada-002"

embedding_a = openai.Embedding.create(input=text_a, model=model_id)['data'][0]['embedding']

embedding_b = openai.Embedding.create(input=text_b, model=model_id)['data'][0]['embedding']

embedding_c = openai.Embedding.create(input=text_c, model=model_id)['data'][0]['embedding']

print(len(embedding_a))

'''
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])
print(chat_completion)
'''

# Use the t-SNE algorithm to transform high dimensional data into two dimensions

# Calculate the cosine simularity between A, B and C embedding
import numpy as np
from numpy.linalg import norm

A = np.array(embedding_a)
B = np.array(embedding_b)
C = np.array(embedding_c)
  
# compute cosine similarity
cosine = np.dot(A,B)/(norm(A)*norm(B))
print("Cosine Similarity A-B:", cosine)

# compute cosine similarity
cosine = np.dot(A,C)/(norm(A)*norm(C))
print("Cosine Similarity A-C:", cosine)