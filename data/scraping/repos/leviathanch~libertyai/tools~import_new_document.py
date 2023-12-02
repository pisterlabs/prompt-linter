from flask import Flask, render_template, request

from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores.pgvector import (
    PGVector,
    DistanceStrategy,
)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader

from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
    ConversationSummaryMemory,
    ConversationKGMemory,
)
from langchain import PromptTemplate, LLMChain

from LibertyAI import (
    LibertyLLM,
    LibertyEmbeddings,
    get_configuration,
)

loader = TextLoader('/home/leviathan/libertyai/critique_of_interventionism_clean.txt')
#loader = TextLoader('/home/leviathan/libertyai/test.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = LibertyEmbeddings(endpoint="http://libergpt.univ.social/api/embedding")

config = get_configuration()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=config.get('DATABASE', 'PGSQL_SERVER'),
    port=config.get('DATABASE', 'PGSQL_SERVER_PORT'),
    database=config.get('DATABASE', 'PGSQL_DATABASE'),
    user=config.get('DATABASE', 'PGSQL_USER'),
    password=config.get('DATABASE', 'PGSQL_PASSWORD'),
)

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    connection_string=CONNECTION_STRING,
#    distance_strategy = DistanceStrategy.COSINE,
)
