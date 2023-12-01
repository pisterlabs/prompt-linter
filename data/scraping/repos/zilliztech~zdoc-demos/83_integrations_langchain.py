from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.zilliz import Zilliz
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

import os

# Set up arguments

# 1. Set up the name of the collection to be created.
COLLECTION_NAME = 'doc_qa_db'

# 2. Set up the dimension of the embeddings.
DIMENSION = 768

# 3. Set up the cohere api key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 4. Set up the connection parameters for your Zilliz Cloud cluster.
URI = 'YOUR_CLUSTER_ENDPOINT'

# 5. Set up the token for your Zilliz Cloud cluster.
# You can either use an API key or a set of cluster username and password joined by a colon.
TOKEN = 'YOUR_CLUSTER_TOKEN'

# Use the WebBaseLoader to load specified web pages into documents
loader = WebBaseLoader([
    'https://milvus.io/docs/overview.md',
    'https://milvus.io/docs/release_notes.md',
    'https://milvus.io/docs/architecture_overview.md',
    'https://milvus.io/docs/four_layers.md',
    'https://milvus.io/docs/main_components.md',
    'https://milvus.io/docs/data_processing.md',
    'https://milvus.io/docs/bitset.md',
    'https://milvus.io/docs/boolean.md',
    'https://milvus.io/docs/consistency.md',
    'https://milvus.io/docs/coordinator_ha.md',
    'https://milvus.io/docs/replica.md',
    'https://milvus.io/docs/knowhere.md',
    'https://milvus.io/docs/schema.md',
    'https://milvus.io/docs/dynamic_schema.md',
    'https://milvus.io/docs/json_data_type.md',
    'https://milvus.io/docs/metric.md',
    'https://milvus.io/docs/partition_key.md',
    'https://milvus.io/docs/multi_tenancy.md',
    'https://milvus.io/docs/timestamp.md',
    'https://milvus.io/docs/users_and_roles.md',
    'https://milvus.io/docs/index.md',
    'https://milvus.io/docs/disk_index.md',
    'https://milvus.io/docs/scalar_index.md',
    'https://milvus.io/docs/performance_faq.md',
    'https://milvus.io/docs/product_faq.md',
    'https://milvus.io/docs/operational_faq.md',
    'https://milvus.io/docs/troubleshooting.md',
])

docs = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
connection_args = { 'uri': URI, 'token': TOKEN }

vector_store = Zilliz(
    embedding_function=embeddings, 
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    drop_old=True,
).from_documents(
    all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args=connection_args,
)

query = "What are the main components of Milvus?"
docs = vector_store.similarity_search(query)

print(len(docs))

# Output
#
# 4



llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) 
retriever = vector_store.as_retriever()

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)

print(rag_chain.invoke("Explain IVF_FLAT in Milvus."))

# Output
#
# content='IVF_FLAT is an index mechanism in Milvus that divides a vector space into clusters. It compares the distances between a target vector and the centers of all clusters to find the nearest clusters. Then, it compares the distances between the target vector and the vectors in the selected clusters to find the nearest vectors. IVF_FLAT demonstrates performance advantages when the number of vectors exceeds the value of nlist. Thanks for asking!'

