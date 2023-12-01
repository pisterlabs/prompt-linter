from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from chromaviz import visualize_collection

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./memory", embedding_function=embedding_function)

visualize_collection(db._collection)
