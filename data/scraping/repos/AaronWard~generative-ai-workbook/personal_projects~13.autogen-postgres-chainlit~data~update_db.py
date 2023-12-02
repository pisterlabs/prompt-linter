"""
This script contains helper functions for handling the
splitting and picking the random chunks of text to generate
questions from. There is also a function for using an LLM
to create an evaluation dataset for us.

"""
import os
import random
import datetime
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import QAGenerationChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _pick_random_chunks(num_qa_pairs: int, max_index: int) -> list:
    """Pick N random sections of split documents to generate questions from."""
    return random.sample(range(0, max_index + 1), num_qa_pairs)

def _split_document_into_chunks(doc_file: str, chunk_size: int = 3000, chunk_overlap: int = 250) -> list:
    """Split the document into equal-sized chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap, 
                                                   length_function=len)
    loader = PyPDFLoader(doc_file)
    return loader.load_and_split(text_splitter=text_splitter)
