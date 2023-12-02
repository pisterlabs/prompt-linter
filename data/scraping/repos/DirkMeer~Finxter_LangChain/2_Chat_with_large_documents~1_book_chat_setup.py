import re
import pinecone
from decouple import config
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Pinecone

embeddings_api = OpenAIEmbeddings(openai_api_key=config("OPENAI_API_KEY"))

pinecone.init(api_key=config("PINECONE_API_KEY"), environment="gcp-starter")
pinecone_index = "langchain-vector-store"

loader = PyPDFLoader("data/How-to-succeed.pdf")
data: list[Document] = loader.load_and_split()
page_texts: list[str] = [page.page_content for page in data]

page_texts_fixed: list[str] = [re.sub(r"\t|\n", " ", page) for page in page_texts]

vector_database = Pinecone.from_texts(
    page_texts_fixed, embeddings_api, index_name=pinecone_index
)
