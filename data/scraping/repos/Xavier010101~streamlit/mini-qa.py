import openai
import os
from dotenv import load_dotenv
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset
load_dotenv()


astra_db_keyspace = "vector_db"
astra_db_bundle_path = f'./secure-connect-vector-database.zip'
client_id = os.getenv("clientId")
secret = os.getenv("secret")
token = os.getenv("token")
bundle = os.getenv("bundle")
openai.api_key = os.getenv('OPENAI_API_KEY')

cloud_config = {
    'secure_connect_bundle': astra_db_bundle_path
}
auth_provider = PlainTextAuthProvider(client_id, secret)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

llm = OpenAI()
myEmbedding = OpenAIEmbeddings()
myCassandraVector = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=astra_db_keyspace,
    table_name="vector_table",
)
print("loding data from hf")

# This code loads the dataset from the Onion News and saves the first 50 headlines into a list called headlines.

# myDataSet = load_dataset("Biddls/Onion_News", split="train")333333333333333333


vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVector)


first_question = True
while True:
    if first_question:
        query_text = input('\nEnter your question (or type "quit" to exit ): ')
    else:
        query_text = input(
            "/nWhat is your next questions (or type 'quit' to exit :)")
    if query_text == 'quit':
        break

    print('QUESTION: \"%s\"' % query_text)
    answer = vectorIndex.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print('DOCUMENTS BY REVELANCE:')
    for doc, score in myCassandraVector.similarity_search_with_score(query_text, k=4):
        print("%0.4f %s...\"" % (score, doc.page_content[:1000]))

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import CassandraVectorStore
# from langchain.vectorstores.index import VectorStoreIndexWrapper
# from cassandra.cluster import Cluster
# from cassandra.auth import PlainTextAuthProvider
# import openai

# # Cassandra database connection
# auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
# cluster = Cluster(['127.0.0.1'], auth_provider=auth_provider)
# session = cluster.connect()

# # Create OpenAI embedding model
# embedding_model = OpenAIEmbeddings()

# # Embed documents into vectors
# documents = ["Hello world", "My name is John"]
# embedded_docs = embedding_model.embed_documents(documents)

# # Save embedded vectors to Cassandra
# vector_store = CassandraVectorStore(session, "mykeyspace", "mytable")
# vector_store.insert_documents(embedded_docs)

# # Wrap Cassandra store in a vector store index
# wrapped_store = VectorStoreIndexWrapper(vector_store)

# # Initialize OpenAI chat model
# openai.api_key = "YOUR_API_KEY"
# chat_model = openai.ChatCompletion()

# # Converse with database
# user_input = "What documents are in the database?"
# chat_response = chat_model.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant conversing with a vector database"},
#     {"role": "user", "content": user_input},
#     {"role": "assistant", "content": wrapped_store.get_document_count()},
#   ]
# )
# print(chat_response['choices'][0]['message']['content'])
