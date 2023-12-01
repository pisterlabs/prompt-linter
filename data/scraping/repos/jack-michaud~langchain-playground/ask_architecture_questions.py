from dotenv import load_dotenv

load_dotenv()
import os

import pinecone
from langchain.agents import tool
from langchain.chains import VectorDBQA, VectorDBQAWithSourcesChain
from langchain.document_loaders import PagedPDFSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI, OpenAIChat
from langchain.vectorstores import Pinecone

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)


# For loading
# loader = PagedPDFSplitter(
#     "/home/jack/Vault/devnotes/assets/The_Software_Architect_Elevator_1676902724666_0.pdf"
# )
# index = VectorstoreIndexCreator(vectorstore_cls=Pinecone).from_loaders([loader])
# index.query("What is a software architect?")

# For evaluating
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(
    pinecone.index.Index("cca9db8e-df72-4ae7-afae-5b73ff0b3800"),
    embedding_function=embeddings.embed_query,
    text_key="text",
)
chain = VectorDBQAWithSourcesChain.from_chain_type(
    OpenAIChat(temperature=0.1), vectorstore=vectorstore
)


@tool("ask_architecture_knowledge")
def ask_architecture_knowledge(query: str) -> str:
    """Ask for general architecture knowledge from The Software Architect Elevator. Specific questions should ask Jack."""
    answer = chain({"question": query})["answer"]

    return f"Architecture said: {answer}"


if __name__ == "__main__":
    while True:
        i = input("You: ")
        print(chain({"question": i})["answer"])
