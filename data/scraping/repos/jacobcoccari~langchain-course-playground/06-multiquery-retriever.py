# Build a sample vectorDB
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

# VectorDB
embedding_function = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base",
)

db = Chroma(
    persist_directory="./10-Retriver-Augmented-Generation/crash-course-db",
    embedding_function=embedding_function,
)


question = "What are the major accomplishments of Ghengis Khan?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=llm,
)

result = retriever_from_llm.get_relevant_documents(query=question)
print(result)
