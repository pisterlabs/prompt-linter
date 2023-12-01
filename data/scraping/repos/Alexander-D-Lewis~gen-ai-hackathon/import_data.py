# std
import os
import pickle
from glob import iglob

# third party
import pandas as pd
import numpy as np
from langchain.document_loaders import UnstructuredODTLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")



def fix_for_nltk_error():
    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # you'll want to download punkt and perceptron_tagger
    nltk.download()



def load_docs():
    docs = []
    
    for file_path in tqdm(iglob("downloaded_docs/downloaded_docs/*.odt")):
        try:
            docs.extend(UnstructuredODTLoader(file_path=file_path).load())

        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    return docs


def split_docs(docs):
    text_splitter = CharacterTextSplitter()

    corpus_list = []

    for doc in tqdm(docs):
            splits = text_splitter.split_text(doc.page_content)
            corpus_list.extend(splits)


    pickle.Pickler(open("corpus_list.pkl", "wb")).dump(corpus_list)
    return corpus_list



if __name__ == "__main__":
    docs = load_docs()
    corpus_list = split_docs(docs)
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)
    breakpoint()
    store = FAISS.from_texts(corpus_list, embedding=hf)
    store.save_local("faiss_index")


    breakpoint()


