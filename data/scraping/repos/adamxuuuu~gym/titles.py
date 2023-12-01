import os
import re
import string
import numpy as np
from tqdm import tqdm
from pprint import pprint

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from config import EMBEDDING

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING,
    encode_kwargs = {'normalize_embeddings': True}
)
files = os.listdir('data/policies')

def find_num(text):
    match = re.search(r'po\D*(\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return 0

def replace_punctuation_with_space(text):
    translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translation_table)

def clean_text(text):
    result = replace_punctuation_with_space(text)
    # Remove consecutive white spaces
    result = re.sub(r"\s+", " ", result)
    return result

def embed_filenames():
    meta = [{"source" : f, "number" : find_num(f)} for f in files]
    faiss = FAISS.from_texts(
        texts=[clean_text(f) for f in files], 
        embedding=embeddings, 
        metadatas=meta
    )
    faiss.save_local('faiss/filenames')

if __name__ == "__main__":
    # find_num('VCIC_Policy No 58-IT Steering V2.0.pdf')
    # embed_filenames()

    faiss = FAISS.load_local('faiss/filenames', embeddings)
    while True:
        query = input("Question:")
        if not query:
            break
        res = faiss.similarity_search_with_score(query, k=10, filter={"number": query}, fetch_k=400)
        # res = faiss.max_marginal_relevance_search(query, k=5, filter={"number": query}, fetch_k=40)
        pprint(res)
