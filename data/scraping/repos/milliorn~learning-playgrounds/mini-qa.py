#!/usr/bin/env python3

# Import necessary libraries
import os
from dotenv import load_dotenv
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from datasets import load_dataset

# Load environment variables from a file named .env
load_dotenv()

# Access your stored environment variables
# These are typically sensitive information, like API keys or database credentials
# They are stored securely in a file called .env
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_CLIENT_ID = os.getenv("ASTRA_DB_CLIENT_ID")
ASTRA_DB_CLIENT_SECRET = os.getenv("ASTRA_DB_CLIENT_SECRET")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
ASTRA_DB_SECURE_BUNDLE_PATH = os.getenv("ASTRA_DB_SECURE_BUNDLE_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration for connecting to the database
cloud_config = {
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}

# Create an authentication provider using your credentials
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)

# Connect to the Cassandra database cluster
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

# Create an instance of the OpenAI language model
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

# Create an instance of OpenAI's embeddings
myEmbeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Set the table name for Cassandra
table_name = "mini_qa"

# Create a Cassandra instance for storing data
myCassandraVStore = Cassandra(
    embedding=myEmbeddings,
    keyspace=ASTRA_DB_KEYSPACE,
    session=astraSession,
    table_name=table_name
)

# Load a dataset named "Biddls/Onion_News" and select the first 50 headlines
print("Loading dataset...")
dataset = load_dataset("Biddls/Onion_News", split="train")
headlines = dataset["text"][:50]

# Store the headlines in the Cassandra database
print("\nGenerating embeddings and storing in AstraDB...")
myCassandraVStore.add_texts(headlines)

# Print the number of headlines inserted
print("Inserted %i headlines.\n" % len(headlines))

# Create a vector store index wrapper for similarity search
vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

# Initialize a flag for the first question
first_question = True

# Start an interactive loop for asking questions
while True:
    if first_question:
        query_text = input("\nAsk a question (or type 'quit' to exit): ")
        first_question = False
    else:
        query_text = input("\nAsk another question (or type 'quit' to exit): ")

    if query_text == "quit":
        break

    print("Question: \"%s\"" % query_text)
    
    # Query the vector index to find the most relevant answer
    answer = vectorIndex.query(query_text, llm=llm).strip()
    
    # Print the answer
    print("Answer: \"%s\"\n" % answer)

    print("Document by relevance:")
    
    # Perform a similarity search and display the most relevant documents
    for doc, score in myCassandraVStore.similarity_search_with_score(query_text, k=4):
        print(" %0.4f %s ...\"" % (score, doc.page_content[:60]))
