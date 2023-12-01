from langchain.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
pdf_path = os.path.abspath("100-years-of-solitude.pdf")


loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(len(pages))

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

index_name = "langchain"


