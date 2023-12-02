from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from dotenv import load_dotenv, find_dotenv
from cqlsession import getCQLSession

from langchain.document_transformers import Html2TextTransformer


import os

load_dotenv(find_dotenv(), override=True)
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]

#Globals
cqlMode = 'astra_db'
table_name = 'vs_4seasons_openai'

session = getCQLSession(mode=cqlMode)

embedding = OpenAIEmbeddings()

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Cassandra,
    embedding=embedding,
    vectorstore_kwargs={
        'session': session,
        'keyspace': ASTRA_DB_KEYSPACE,
        'table_name': table_name,
    },
)

# Load HTML
# loader = AsyncChromiumLoader(["https://www.fourseasons.com/austin/"])
loader = AsyncHtmlLoader(["https://www.fourseasons.com/austin/"])
html = loader.load()

# // div class="normal
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["div"])
# print(docs_transformed[0].page_content[0:500])

index_creator.from_documents(docs_transformed)


