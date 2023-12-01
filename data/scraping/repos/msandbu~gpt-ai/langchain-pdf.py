# pip install langchain openai google-search-results

import os
from dotenv import load_dotenv
import openai
import langchain
import getpass

os.environ["OPENAI_API_KEY"] ="OPENAPIKEY"

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

loader = PyPDFLoader("PDFFILE.PDF")
pages = loader.load_and_split()
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])