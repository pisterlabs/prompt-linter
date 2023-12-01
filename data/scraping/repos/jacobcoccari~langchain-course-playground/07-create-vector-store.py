from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

loader = TextLoader(
    "./09-chunking-embeddings-and-vector-stores/jfk-inaguration-speech.txt"
)
speech = loader.load()[0].page_content


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=50,
)

metadatas = [{"title": "JFK Inauguration Speech", "author": "John F. Kennedy"}]

texts_with_metadata = text_splitter.create_documents([speech], metadatas=metadatas)

for doc in texts_with_metadata:
    print(len(doc.page_content))

from langchain.embeddings import HuggingFaceInstructEmbeddings

embedding_function = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base",
)

db = Chroma.from_documents(
    texts_with_metadata,
    embedding_function,
    persist_directory="./09-chunking-embeddings-and-vector-stores/speech-embeddings-db",
)

db.persist()
