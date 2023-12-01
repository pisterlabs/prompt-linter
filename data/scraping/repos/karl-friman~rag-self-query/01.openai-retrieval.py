# chatgpt.py

# External imports
import os
import sys

# Internal module imports
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import constants
from langchain.vectorstores import Chroma

# Set API Key from constants
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# Configuration: whether to save the model to disk & reuse
PERSIST = False

# Check if a query was passed as an argument
query = sys.argv[1] if len(sys.argv) > 1 else None

# Load existing model from disk or create a new one
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(
        persist_directory="persist", embedding_function=OpenAIEmbeddings()
    )
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # For loading data from a directory (in this case it will load the name of a cat from the pdf and the a dog from the txt):
    loader = DirectoryLoader("data/")
    # For loading data from a single file:
    # loader = TextLoader("data/dog.txt")
    index_creation_args = {"persist_directory": "persist"} if PERSIST else {}
    index = VectorstoreIndexCreator(**index_creation_args).from_loaders([loader])

# Initialize the retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

print(
    "Try: 'what is my cat's name?' to read from the retrieval system. Default is reading from the PDF."
)

# Main chat loop
chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ["quit", "q", "exit"]:
        sys.exit()

    result = chain({"question": query, "chat_history": chat_history})
    print(result["answer"])

    chat_history.append((query, result["answer"]))
    query = None
