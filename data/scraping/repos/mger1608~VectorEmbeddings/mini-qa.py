ASTRA_DB_SECURE_BUNDLE_PATH = "INSERT KEY HERE"
ASTRA_DB_APPLICATION_TOKEN = "INSERT KEY HERE"
ASTRA_DB_CLIENT_ID = "INSERT KEY HERE"
ASTRA_DB_CLIENT_SECRET = "INSERT KEY HERE"
ASTRA_DB_KEYSPACE_NAME = "INSERT KEY HERE"
OPEN_API_KEY = "INSERT KEY HERE"

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT, ProtocolVersion
...
profile = ExecutionProfile(request_timeout=30)
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

cloud_config= {
'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider, protocol_version=ProtocolVersion.V4, execution_profiles={EXEC_PROFILE_DEFAULT: profile})
astraSession = cluster.connect()

llm = OpenAI(openai_api_key=OPEN_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

myCassandraVStore = Cassandra(
    embedding = myEmbedding,
    session = astraSession,
    keyspace = ASTRA_DB_KEYSPACE_NAME,
    table_name = "qa_mini_demo",
)

print("loading data from huggingface")
myDataset = load_dataset("Biddls/Onion_News", split = "train")
headlines = myDataset["text"][:50]

print("\nGenerating embeddings and storing in AstraDB")
myCassandraVStore.add_texts(headlines)

print("Inserted %i headlines.\n" % len(headlines))

vectorIndex = VectorStoreIndexWrapper(vectorstore = myCassandraVStore)

first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ")
        first_question = False
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ")

    if query_text.lower() == 'quit':
        break

    print("QUESTION: \"%s\"" % query_text)
    answer = vectorIndex.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in myCassandraVStore.similarity_search_with_score(query_text, k=4):
        print("  %0.4f \"s ...\"" % (score, doc.page_content[:60]))