from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

print("Loading files...")
loader = DirectoryLoader(
    path='./sources', 
    glob="**/*.pdf", 
    show_progress=True,
    loader_cls=PyMuPDFLoader,
    recursive=True
    )
docs = loader.load()

if len(docs) == 0:
    print("No docs loaded!")
    exit()

print("Splitting into chunks...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs_split = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Storing into Chroma, this can take a long time...")
ids = [str(i) for i in range(1, len(docs_split) + 1)]
db = Chroma.from_documents(docs_split, embedding_model, persist_directory="./chroma_db", ids=ids)

print("DONE!")
