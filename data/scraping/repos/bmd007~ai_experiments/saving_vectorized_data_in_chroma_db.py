from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

persist_directory = '../../local_data/vector_db_persistence'
loader = JSONLoader(file_path="../../local_data/small.json",
                    jq_schema=".[]",
                    text_content=False)
documents = loader.load()
embedding = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=800)
docs = text_splitter.split_documents(documents)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
