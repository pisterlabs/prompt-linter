from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

from src.utils import get_config

# Get configs
cfg = get_config()

# Load PDF file from data path
loader = DirectoryLoader(cfg.DATA_PATH, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split text from PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP
)
texts = text_splitter.split_documents(documents)

# Load embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=cfg.EMBEDDINGS_MODEL_NAME,
    model_kwargs={"device": "cuda" if cfg.USE_GPU else "cpu"},
)

# Build and persist FAISS vector store
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local(cfg.DB_FAISS_PATH)
