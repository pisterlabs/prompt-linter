from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from utils import load_documents, load_db, save_db, load_embeddings
from dotenv import load_dotenv

load_dotenv()

db = load_db(embedding_function=load_embeddings())
db.add_documents(load_documents("new_document/"))
save_db(db)