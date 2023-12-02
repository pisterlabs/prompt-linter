import os
ASTRA_DB_SECURE_BUNDLE_PATH = r"C:\Users\yoga gen 5\Documents\Chat'innov\Python_env\search-python\secure-connect-vector-database.zip"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:pzzDNsgbRwyFOgXbEhedKbjk:0194015953a28fffb6378e6809e1aac51a9656afecde55f2aef5401b0d243020"
ASTRA_DB_CLIENT_ID = "pzzDNsgbRwyFOgXbEhedKbjk"

ASTRA_ID_CLIENT_SECRET = "RFAljaAWW,J9fBxF.HXRrlLukbITX7CqqF,ZZ-Iz4Eq,LB-leL,KNIjOrr1g+uamEXs3k1Lezd1JknpfbOX11jZ8m7T5DF2WlLh_9m-3R-q7CUbYiw092S6C64.y+F0z"
ASTRA_DB_KEYSPACE = "search"
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

cloud_config = {
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_ID_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider = auth_provider)
astraSession = cluster.connect()

llm = OpenAI(openai_api_key = OPEN_AI_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key = OPEN_AI_KEY)

myCassandraVStore = Cassandra(
    embedding= myEmbedding,
    session= astraSession,
    keyspace= ASTRA_DB_KEYSPACE,
    table_name= "qa_mini_demo",
)

print("Loading my data from huggingface")
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset["text"][:50]

print("\nGenerating embeddings and storing in AstraDB")
myCassandraVStore.add_texts(headlines)

print("Inserted %i headlines.\n" %len(headlines))

vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

first_question = True 
while True:
    if first_question:
        querry_text = input("\nEnter your question (or type 'quit' to exit): ")
        first_question = False
    else:
        querry_text = input("\nWhat's your next question (or type 'quit' to exit): ")
    if querry_text.lower() == 'quit':
        break

    print("QUESTION: \"%s\"" %querry_text)
    answer = vectorIndex.query(querry_text, llm=llm).strip()
    print("ANSWER: \"%s\"" %answer)

    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in myCassandraVStore.similarity_search_with_score(querry_text, k=4):
        print(" %0.4f \"%s...\"" %(score, doc.page_content[:60]))