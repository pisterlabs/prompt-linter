from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DeepLake
import os

FILE = "./Manual.pdf"

os.environ["OPENAI_API_KEY"] = ''
os.environ['ACTIVELOOP_TOKEN'] = ""

# Example example below. Switch from curl to wget if using Linux
# !curl -LO https://github.com/activeloopai/examples/raw/main/colabs/starting_data/paul_graham_essay.txt --output "paul_graham_essay.txt"
# source_text = "paul_graham_essay.txt"

dataset_path = "hub://agarg/formula_manual"

loader = PyPDFLoader(FILE)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

db = DeepLake.from_documents(docs, dataset_path=dataset_path, embedding=OpenAIEmbeddings())