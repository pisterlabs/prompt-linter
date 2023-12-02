# pineconeにデータを追加する

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os


pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")
index = pinecone.Index("rag")
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(index=index, embedding=embeddings, text_key="text")

tools = """
Tools are a new way to interact with OpenAI’s powerful models.
Give Assistants access to OpenAI-hosted tools like Code Interpreter and Knowledge Retrieval,
 or build your own tools using Function calling. 
 Usage of OpenAI-hosted tools comes at an additional fee — visit our help center article to learn more 
 about how these tools are priced.
"""

code_interpreter = """
Code Interpreter allows the Assistants API to write and run Python code in a sandboxed execution environment. This tool can process files with diverse data and formatting, and generate files with data and images of graphs. Code Interpreter allows your Assistant to run code iteratively to solve challenging code and math problems. When your Assistant writes code that fails to run, it can iterate on this code by attempting to run different code until the code execution succeeds.
"""

# データの追加
vectorstore.add_texts(texts=[tools], metadatas=[{"title": "tools"}], embedding_chunk_size= 1000)

vectorstore.add_texts(texts=[code_interpreter], metadatas=[{"title": "code_interpreter"}], embedding_chunk_size= 1000)