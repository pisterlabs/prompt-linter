import json
import re
import hashlib


from aiohttp import ClientSession

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.utilities import SearxSearchWrapper

from langchain.tools import BaseTool

def get_vector_db_tool():
    config = get_configuration()
    # DB Vectors in PostgreSQL:
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=config.get('DATABASE', 'PGSQL_SERVER'),
        port=config.get('DATABASE', 'PGSQL_SERVER_PORT'),
        database=config.get('DATABASE', 'PGSQL_DATABASE'),
        user=config.get('DATABASE', 'PGSQL_USER'),
        password=config.get('DATABASE', 'PGSQL_PASSWORD'),
    )
    embeddings = LibertyEmbeddings(
        endpoint = "https://libergpt.univ.social/api/embedding"
    )
    db = PGVector(
        embedding_function = embeddings,
        connection_string = CONNECTION_STRING,
    )
    return Tool(
        name = "PGVector",
        func=db.similarity_search_with_score,
        description="useful for when you need to look up context in your database of reference texts."
    )

    #tools = []
    #tools.append(get_date_time_tool())
    #tools.append(get_vector_db_tool())
    tools += load_tools(
        ["searx-search"], searx_host="http://libergpt.univ.social/searx",
        llm = LibertyLLM(
            endpoint = "https://libergpt.univ.social/api/generation",
            temperature = 0,
            max_tokens = 20,
            verbose = True,
        ),
    )
