import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.document_loaders import CSVLoader
from dotenv import load_dotenv

def vectorize(data_path):
    load_dotenv()

    # Generate Embeddings
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name = "text-embedding-ada-002"
    )
    
    # Initialize ChromaDB & Create Collection
    chroma_db = chromadb.PersistentClient(path="vectordb")
    collection = chroma_db.get_or_create_collection(name="rfp_generator", embedding_function=openai_ef)

    # Load Data
    loader = CSVLoader(data_path)
    data = loader.load()

    # Add Embeddings to Collection
    docs = [doc.page_content.replace("\n", " ") for doc in data[:10000]]
    metadatas = [doc.metadata for doc in data[:10000]]
    ids = [str(id) for id in range(1, len(data[:10000])+1)]

    collection.add(
        documents=docs, metadatas=metadatas, ids=ids
    )
    
    return collection

# collection = vectorize('Dataset-small.csv')
# print (collection.peek())