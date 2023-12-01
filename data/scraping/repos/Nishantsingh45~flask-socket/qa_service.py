# Import necessary libraries and set up Pinecone integration (as in your existing code)
import os
import pinecone
import nest_asyncio
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

# Initialize Pinecone
pinecone.init(api_key="d054afe9-f7a9-4064-a940-405c9b96602f", environment="gcp-starter")

# Fix a bug with asyncio and Jupyter
nest_asyncio.apply()

# Load documents, split them into chunks, and create Pinecone index
loader = SitemapLoader("https://bestbuzzidea.blogspot.com/sitemap.xml")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, length_function=len)
docs_chunks = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
index_name = "ind"
#docsearch = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name)
docsearch = Pinecone.from_existing_index(index_name, embeddings)
# Create a retrieval QA chain
llm = OpenAI()
qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

# Define a query function
def query_document(query):
    result = qa_with_sources({"query": query})
    return result["result"]
