from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\VectorStorage\Text.pdf")
pages = loader.load()

chunks = []
overlap_percentage = 0.06
chunk_size_percentage = 0.5

for page in pages:
    overlapping_strings = []
    page_length = len(page.page_content)

    overlap_size = int(page_length * overlap_percentage)

    chunk_size = int(page_length * chunk_size_percentage)

    for i in range(0, page_length - overlap_size + 1, int(0.9 * chunk_size)):
        substring = page.page_content[i:i + chunk_size + overlap_size]
        overlapping_strings.append(substring)

    chunks.extend(overlapping_strings)

import openai

openai.api_key = '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'
embd = []
for chunk in chunks:
    response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
    embd.append(response)

query = "What is NDA?"
response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
query_emb = response['data'][0]['embedding']
query_emb = np.array(query_emb, dtype=np.float32)

import faiss
import numpy as np

embeddings = [emb['data'][0]['embedding'] for emb in embd]
embeddings = np.array(embeddings, dtype=np.float32)

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)

index.add(embeddings)

faiss.write_index(index, 'embedding_index.faiss')

query_embedding = np.array([query_emb])
k = 5
distances, indices = index.search(query_embedding, k)
indices = indices[0]

context = ''
for indice in indices:
    context = context + ' ' + chunks[indice]

messages = [
    {'role': 'system',
     'content': f"Use only the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say 'thanks for asking!' at the end of the answer. \n {context} "},
    {'role': 'user', 'content': f"{query}"}
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0,
)

response['choices'][0]['message']['content']

