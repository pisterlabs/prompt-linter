from pathlib import Path
from typing import List, Tuple

from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field
from langchain.chains import ConversationalRetrievalChain
import os
# Constants
from dotenv import load_dotenv

# load .env file
dotenv_path = '.env'  # Assuming the .env file is in the same directory as your script
load_dotenv(dotenv_path)


llama_embeddings_model = os.getenv("LLAMA_EMBEDDINGS_MODEL")
model_type = os.getenv("MODEL_TYPE")
model_path = os.getenv("MODEL_PATH")
model_n_ctx = os.getenv("MODEL_N_CTX")
openai_api_key = os.getenv("OPEN_AI_API_KEY")
index_path = os.getenv("INDEX_PATH")
text_path = os.getenv("TEXT_PATH")

# Functions
def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=llama_embeddings_model)

def load_documents() -> List:
    loader = TextLoader(text_path)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Main execution
# llm = GPT4All(model=local_path, n_ctx=2048, verbose=True)
llm = OpenAI(openai_api_key=openai_api_key)

embeddings = initialize_embeddings()
# sources = load_documents()
# chunks = split_chunks(sources)
# vectorstore = generate_index(chunks, embeddings)
# vectorstore.save_local("full_sotu_index")

index = FAISS.load_local(index_path, embeddings)

qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=400)

# Chatbot loop
chat_history = []
print("Welcome Agri-mche chatbot! Type 'q' to stop.")
while True:
    query = input("Please enter your question: ")
    
    if query.lower() == 'q':
        break
    result = qa({"question": query, "chat_history": chat_history})

    print("Answer:", result['answer'])
