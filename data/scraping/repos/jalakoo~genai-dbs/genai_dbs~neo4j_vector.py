# from https://python.langchain.com/docs/integrations/vectorstores/neo4jvector

import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Neo4jVector

loader = TextLoader("../../modules/state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

url = "bolt://localhost:7687"
username = "neo4j"
password = "pleaseletmein"

# The Neo4jVector Module will connect to Neo4j and create a vector index if needed.
db = Neo4jVector.from_documents(
    docs, OpenAIEmbeddings(), url=url, username=username, password=password
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query, k=2)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

index_name = "vector"  # default index name

store = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
)

store.query("CREATE (p:Person {name: 'Tomaz', location:'Slovenia', hobby:'Bicycle'})")


existing_graph = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    node_label="Person",
    text_node_properties=["name", "location"],
    embedding_node_property="embedding",
)
result = existing_graph.similarity_search("Slovenia", k=1)

result[0]