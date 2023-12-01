
from langchain import PromptTemplate, LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import weaviate
import os
import json

import argparse

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import WeaviateHybridSearchRetriever
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.vectorstores import Weaviate
import uuid
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from langchain import OpenAI, LLMMathChain

import os

embeddings = OpenAIEmbeddings()

from deep_translator import (GoogleTranslator)

def _convert_pdf_to_document(path: str = None):

    """Convert a PDF document to a Document object"""
    if path is None:
        raise ValueError("A valid path to the document must be provided.")

    loader = PyPDFLoader(path)
    pages = loader.load_and_split()

    print("PAGES", pages[0])

    # Parse metadata from the folder path
    path_parts = Path(path).parts
    arch_docs = path_parts.index("arch_docs")
    metadata_parts = path_parts[arch_docs+1:]

    documents = []
    for page in pages:
        translation = GoogleTranslator(source='auto', target='en').translate(text=page.page_content)
        documents.append(
            Document(
                metadata={
                    "title": "Personal Receipt",
                    "country": metadata_parts[1],
                    "year": metadata_parts[0],
                    "author": str(uuid.uuid4()),
                    "source": "/".join(metadata_parts),
                },
                page_content=translation,
            )
        )
    print(documents)

    return documents



def _init_weaviate():
    """Initialize weaviate client and retriever"""
    auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
    client = weaviate.Client(
        url='https://my-vev-index-o4qitptw.weaviate.network',
        auth_client_secret=auth_config,

        additional_headers={
            "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
        }
    )
    retriever = WeaviateHybridSearchRetriever(
        client=client,
        index_name="Architecture",
        text_key="text",
        attributes=[],
        embedding=embeddings,
        create_schema_if_missing=True,
    )

    return retriever
def load_to_weaviate(document_path=None):
    """Load documents to weaviate"""
    retriever =_init_weaviate()

    docs = _convert_pdf_to_document(document_path)

    return retriever.add_documents(docs)