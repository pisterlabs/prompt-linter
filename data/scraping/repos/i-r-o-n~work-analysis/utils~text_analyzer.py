import os
import streamlit as st

from utils.file_manager import get_dataset_text
from utils.data_types import Datasets, ModelTypes, Dataset, Model

# load api secrets
key = st.secrets.api_key
os.environ["OPENAI_API_KEY"] = key

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA



class Defaults:
    temperature = 0.7,
    model = ModelTypes.MAP_REDUCE.value


def make_query(
        query: str, 
        dataset: Dataset = Datasets.ALL.value, 
        temperature: float = Defaults.temperature, 
        model: Model = ModelTypes.MAP_REDUCE.value) -> str:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0) 
    texts = splitter.split_text(get_dataset_text(dataset))
    embeddings = OpenAIEmbeddings()
    search_index = Chroma.from_texts(texts, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=temperature), 
        chain_type=model, retriever=search_index.as_retriever())

    return qa.run(query)



