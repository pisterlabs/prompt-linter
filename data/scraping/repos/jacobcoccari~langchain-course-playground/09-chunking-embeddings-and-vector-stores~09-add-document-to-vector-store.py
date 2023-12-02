from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

loader = TextLoader(
    "./09-chunking-embeddings-and-vector-stores/jfk-rice-university-speech.txt"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=50,
)
speech = loader.load()[0].page_content


metadatas = [{"title": "JFK Rice University Speech", "author": "John F. Kennedy"}]

texts_with_metadata = text_splitter.create_documents(
    [speech],
    metadatas=metadatas,
)

embedding_function = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base",
)

db = Chroma.from_documents(
    texts_with_metadata,
    embedding_function,
    persist_directory="./09-chunking-embeddings-and-vector-stores/speech-embeddings-db",
)

db.persist()

docs = db.similarity_search("we choose to go to the moon", k=2)
print(docs)

question = "what did jfk let every nation know during his inaguration speech?"
docs = db.similarity_search(question, k=2)
print(docs)
