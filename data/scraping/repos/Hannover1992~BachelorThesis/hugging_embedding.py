from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from langchain.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer

loader = TextLoader('all_txt.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
docs = text_splitter.split_documents(documents)



# embeddings = OpenAIEmbeddings()
# embeddings = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

db = Chroma.from_documents(docs, embeddings, persist_directory='HuggingFaceDB2')

query = "Wissenschaftliche Ghostwriter"
docs = db.similarity_search(query)

for doc in docs:
    print(doc)


