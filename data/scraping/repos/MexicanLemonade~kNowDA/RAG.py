import langchain

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import SpacyTextSplitter, NLTKTextSplitter, CharacterTextSplitter
from langchain.schema import Document

import cohere
import os
import re
import textract
import numpy as np
from typing import List

def set_key(key):
    os.environ['COHERE_API_KEY'] = key
    # co = cohere.Client(key)

def pdf_to_chunks(loader: PyPDFLoader):
    # loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    chunked_pages = []
    for page in pages:
        chunked_page_content = page.page_content.split('\n\n')
        for chunk in chunked_page_content:
            chunked_pages.append(Document(page_content=chunk, metadata=page.metadata))
    return chunked_pages

def doc_generate(question, pdf_loader):
    docs = [dict(chunk) for chunk in pdf_to_chunks(pdf_loader)]
    # print(docs[0].keys())
    co = cohere.Client(os.environ['COHERE_API_KEY'])
    response = co.chat(
        chat_history=[
            {"role": "USER", "message": "Take a deep breath and think step by step.\
             Act as an experienced assistant who is helping people understanding the NDA.\
             Given a user statement or question, cite the relevant clauses in the NDA\
             and provide a short explanation."},
        ],
        message=question,
        #   connectors=[{"id": "web-search"}] # perform web search before answering the question
        documents=[{'snippet': doc['page_content']} for doc in docs]
        , 
        prompt_truncation='AUTO'
    )
    return response

def nltk_chunking(doc: str):
    import nltk
    # check if spacy model is installed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    chunks = nltk.tokenize.sent_tokenize(doc)
    # print(chunks.sents)

    return list(chunks)

def sim_search(query: str, collection: List[str], threshold=0.8):
    embedding = CohereEmbeddings()
    gen_sent_embedding = embedding.embed_query(query)
    chunk_embeddings = embedding.embed_documents([str(text) for text in collection])
    similarities = np.dot(gen_sent_embedding/np.linalg.norm(gen_sent_embedding), np.array(chunk_embeddings/np.linalg.norm(chunk_embeddings)).T)
    max_idx = np.argmax(similarities)
    if similarities[max_idx] > threshold:
        return collection[max_idx]
    else:
        return None

if __name__ == '__main__':
    question = "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement."
    doc_name = "01_Bosch-Automotive-Service-Solutions-Mutual-Non-Disclosure-Agreement-7-12-17.pdf"
    print(doc_generate(question, doc_name))