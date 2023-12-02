from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings


embedding_function = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base",
)

db_connection = Chroma(
    persist_directory="./09-chunking-embeddings-and-vector-stores/speech-embeddings-db",
    embedding_function=embedding_function,
)
# Connection has been successfully established.
# print(db_connection)

question = "what did jfk let every nation know during his inaguration speech?"
document = db_connection.similarity_search_with_score(question, k=3)
print(document)
