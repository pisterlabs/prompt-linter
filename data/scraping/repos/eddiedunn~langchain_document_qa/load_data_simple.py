import sys
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
#from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from chromadb.config import Settings

from langchain.document_loaders.directory import DirectoryLoader



PERSIST_DIRECTORY = "./db-simple"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

if len(sys.argv) > 1:
    if os.path.isdir(sys.argv[1]):
        # If the argument is a directory, use DirectoryLoader
        loader = DirectoryLoader(sys.argv[1], glob="**/*.txt", loader_cls=TextLoader)
    elif os.path.isfile(sys.argv[1]):
        # If the argument is a file, use TextLoader
        loader = TextLoader(sys.argv[1])
    else:
        print("The provided argument is neither a file nor a directory.")
        sys.exit(1)
    documents = loader.load()
else:
    print("Please provide a file or directory to load.")
    sys.exit(1)


# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=325)

# Split your documents into texts
texts = text_splitter.split_documents(documents)

# Turn your texts into embeddings
#embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    # Create embeddings
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl",
    model_kwargs={"device": "cuda"},
)

# Get your docsearch ready
#docsearch = FAISS.from_documents(texts, embeddings)

db = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory=PERSIST_DIRECTORY,
    client_settings=CHROMA_SETTINGS,
)
db.persist()
db = None


