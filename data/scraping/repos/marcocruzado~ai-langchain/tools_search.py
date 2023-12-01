from langchain.docstore.document import Document
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from typing import List, Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models  import ChatOpenAI
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional, Type
import openai
from functions.embeddings_demo import reloadBd
from langchain.tools import BaseTool
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY", "")


# Loading Embeddings
embeddings = OpenAIEmbeddings()


# Loading Database
connection_string = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5433")),
    database=os.environ.get("PGVECTOR_DATABASE", "doc_search"),
    user=os.environ.get("PGVECTOR_USER", "pguser"),
    password=os.environ.get("PGVECTOR_PASSWORD", "password"),
)


# name of the collection in the database
collection_name = "reglas_de_transito"


class BusquedaEnBd(BaseTool):
    name = "BusquedaEnBd"
    description = "Usa esto cuando nesesites buscar informacion sobre multas de transito"
    # definimos el metodo run

    def _run(self, query: str, run_manager=None) -> str:
        reload = reloadBd()
        Retrier = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(),
            retriever=reload.as_retriever())
        return Retrier(
            {"question": query}
        )

    async def _arun(self, query: str, run_manager=None) -> str:
        reload = reloadBd()
        Retrier = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(),
            retriever=reload.as_retriever())
        return Retrier(
            {"question": query}
        )