import openai

ASTRA_DB_SECURE_BUNDLE_PATH = "C:\\Users\\ljfit\\Desktop\\Random Coding\\Vector Database\\search-python\\search-python\\secure-connect-vector-database.zip"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:TmrEZpkvrtdsbZyZOggyZWjk:ed6a3a3cb7aad9366c7ef6262fac6736702ba4c60d4bbd6537db2dde5454f2ee"
ASTRA_DB_CLIENT_ID = "TmrEZpkvrtdsbZyZOggyZWjk"
ASTRA_DB_CLEINT_SECRET = "zd.Ua74DuKhbrYR38vcDXIWRojRtDH,1UkEwjPwyZpZ9hXoiQ2PctSAuzh9K3AZs,eZ0O7P1NN9RzmJFpdsI0.biwemG2m.424uK1DPjt2AgK4Qdouox5FHsPLs-C1mJ"
ASTRA_DB_KEYSPACE = "search"
OPENAI_API_KEY = openai.api_key = "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"

from langchain.vectorstores.cassandra import Cassandra 
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

cloud_config= { 
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}

auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_ID)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

# llms = OpenAI(OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

myCassandraVectorStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="qa_mini",
)

print("loading data from hugging face")
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset["text"][:50]

print("\n Generated embeddings and storing in AstraDB")
myCassandraVectorStore.add_texts(headlines)

print("Inserted %i headlines.\n" % len(headlines))