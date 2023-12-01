
# Update these with your own api keys and paths
ASTRA_DB_SECURE_BUNDLE_PATH="your secure bundle path here"
ASTRA_DB_APPLICATION_TOKEN="Your token here"
ASTRA_DB_CLIENT_ID="Your client id here"
ASTRA_DB_CLIENT_SECRET ="Your client secret here"
ASTRA_DB_KEYSPACE="search"
OPENAI_API_KEY="Your openai api key here"

# import the necessary packages
from langchain.vectorstores.cassandra import Cassandra # Cassandra is a vector store
from langchain.indexes.vectorstore import VectorStoreIndexWrapper # VectorStoreIndexWrapper is a vector index
from langchain.llms import OpenAI # OpenAI is a language model
from langchain.embeddings import OpenAIEmbeddings # OpenAIEmbeddings is an embedding model

from cassandra.cluster import Cluster # Cluster is a cassandra client
from cassandra.auth import PlainTextAuthProvider # PlainTextAuthProvider is a cassandra client

from datasets import load_dataset # load_dataset is a huggingface function

#   Connect to AstraDB
cloud_config= {
        'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

#   Connect to OpenAI
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#   Connect to AstraDB Vector Store
myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="qa_mini_demo",
)

# Load data from huggingface and store in AstraDB
# Default is just 50 headlines for demo purposes
print("Loading data from huggingface")
dataset = load_dataset("Biddls/Onion_News", split="train")
headlines = dataset["text"][:50]

#  Generate embedding and store in AstraDB
print("\nGenerating embedding and storing in AstraDB")
myCassandraVStore.add_texts(headlines)

print("Inserted %i headlines.\n" % len(headlines))

#   Query AstraDB for similar headlines
vectorIdx = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

#  Ask for a headline and return the most similar headlines
first_question = True
while True:
    if first_question:
    
        query_text = input("Ask me a question: (or type 'quit to exit): ")
        first_question = False
    else:
        query_text = input("Ask me another question: (or type 'quit to exit): ")

    if query_text == "quit":
        break
    
    # display the question
    print("QUESTION: \%s\"" % query_text)

    # get the answer from the index
    answer = vectorIdx.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    # get the most similar headlines
    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in myCassandraVStore.similarity_search_with_score(query_text, k=4):
        print("  %0.4f \"%s...\"" % (score, doc.page_content[:60]))
        