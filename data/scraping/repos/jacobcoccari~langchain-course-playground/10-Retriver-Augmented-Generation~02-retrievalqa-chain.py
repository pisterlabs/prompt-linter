from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

embedding_function = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base",
)

db = Chroma(
    persist_directory="./10-Retriver-Augmented-Generation/crash-course-db",
    embedding_function=embedding_function,
)

# We can see that by setting a different k for our retriever, we get a different and more comprehensive result.
retriever = db.as_retriever(search_kwargs={"k": 5})

model = ChatOpenAI()

qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
)

query = "who was ashoka?"
response = qa.run(query)
print(response)
