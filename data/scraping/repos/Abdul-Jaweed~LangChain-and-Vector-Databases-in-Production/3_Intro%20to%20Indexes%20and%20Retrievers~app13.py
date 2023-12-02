import cohere
from langchain.embeddings import CohereEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

cohere_api = os.getenv("COHERE_API_KEY")

# Initialize the CohereEmbeddings object

cohere = CohereEmbeddings(
    cohere_api_key=cohere_api,
    model="embed-multilingual-v2.0"
)

# Define a list of texts

texts = [
    "Hello from Cohere!", 
    "مرحبًا من كوهير!", 
    "Hallo von Cohere!",  
    "Bonjour de Cohere!", 
    "¡Hola desde Cohere!", 
    "Olá do Cohere!",  
    "Ciao da Cohere!", 
    "您好，来自 Cohere！", 
    "कोहेरे से नमस्ते!"
]


# Generate embeddings for the texts

document_embeddings = cohere.embed_documents(texts)

# Printing the embeddings 

for text, embedding in zip(texts, document_embeddings):
    print(f"Text : {text}")
    print(f"Embedding : {embedding[:5]}") # print first 5 dimensions of each embedding