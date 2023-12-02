import os
import openai
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

from utils.custom_pdf_loader import CustomPDFLoader
from app.database.config import settings


load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = os.getenv("OPENAI_API_VERSION")

loader = CustomPDFLoader("../../../docs/gesetzentwurf.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

connection_string = settings.DATABASE_URL()
collection_name = "documents"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=collection_name,
    connection_string=connection_string,
)

query = "Wovon handelt der Gesetzentwurf?"
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
