from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
import os
from configparser import ConfigParser
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


cfg = ConfigParser()
cfg.read('config.cfg')
OPENAI_API_KEY = cfg.get('API_KEYS', 'OPENAI_API_KEY')

loader = PyPDFLoader('darktrace_report.pdf')
pages = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


faiss_index = FAISS.from_documents(pages, embeddings)


def call_chain(query):
  docs = faiss_index.similarity_search(query, k=3)

  chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), chain_type="refine")
  output = chain({'input_documents':docs, 'question':query})

  return output

query = "Did revenue change in fiscal year 2022? If so, by how much?"
res = call_chain(query)