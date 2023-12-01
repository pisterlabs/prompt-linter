import os

os.environ["OPENAI_API_KEY"]

from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
from langchain.agents import AgentType


class DocumentInput(BaseModel):
    question: str = Field()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

tools = []
files = [
    {
        "name": "sunday-report",
        "path": "/var/www/html/serianu_projects/projects/LangChain_StreamLit/Langchain/Sunday.pdf",
    },
    {
        "name": "wednesday-report",
        "path": "/var/www/html/serianu_projects/projects/LangChain_StreamLit/Langchain/wednesday.pdf",
    },
]

for file in files:
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0613",
)

agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)

agent({"input": "what is the difference in the ATM Account Activities of the two reports"})


def MultipleReportsEngine(query):
    date = agent({"input": query})
    return date
