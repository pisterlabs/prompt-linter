import lancedb
from langchain.vectorstores import LanceDB
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
db = lancedb.connect(".lance-data")
path = "/workspace/flancian"
loader = DirectoryLoader(path, glob="**/*.md")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings()
table = db.create_table(
    "journal",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
            "source": "test"
        }
    ],
    mode="overwrite",
)
LanceDB.from_documents(documents, embeddings, connection=table)
