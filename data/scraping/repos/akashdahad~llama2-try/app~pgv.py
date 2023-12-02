
#? MVP to Use PGVector  
# Installations:
# pip3 install pgvector
# pip3 install psycopg2-binary
# pip3 install fastapi


from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
#* Database connection parameters
DBNAME = 'postgres'
USER = 'devadmin'
PASSWORD = 'KCE9MP2En93gLCz2'
HOST = 'bhyve-india-dev-db.postgres.database.azure.com'
PORT = '5432'


loader = TextLoader("./india.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
embed_model = HuggingFaceEmbeddings(model_name=embed_model_id, encode_kwargs={'batch_size': 32})

CONNECTION_STRING = PGVector.connection_string_from_db_params(driver="psycopg2", host=HOST, port=PORT, database=DBNAME, user=USER, password=PASSWORD)

# The PGVector Module will try to create a table with the name of the collection.
# So, make sure that the collection name is unique and the user has the permission to create a table.

COLLECTION_NAME = "COMPANY_1"

db = PGVector.from_documents(
    embedding=embed_model,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    schema_name="bhyve",
    table_name="pg_embed",
)

# vector_store = PGVector(
#     connection_string=CONNECTION_STRING,
#     embedding_function=my_embedding_function,
#     schema_name="bhyve",
#     table_name="pg_embed",
# )

print(db)

query = "What is Indian History"
docs_with_score = db.similarity_search_with_score(query)


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)