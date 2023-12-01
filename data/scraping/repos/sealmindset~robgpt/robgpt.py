import os
import sys
import subprocess

def install_missing_modules():
    required_modules = [
        'langchain',
        'openai',
        'chromadb',
        'tiktoken',
        'unstructured',
        'constants'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            print(f"Installing missing module: {module}...")
            subprocess.run(["pip3", "install", module])
    
    # Special installation for unstructured[pdf]
    try:
        __import__('unstructured')
    except ImportError:
        subprocess.run(["pip3", "install", "unstructured[pdf]"])

install_missing_modules()

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

#pip3 install pip3 install langchain openai chromadb tiktoken unstructured
#pip3 install constants
#pip3 install "unstructured[pdf]"

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

try:
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

except ImportError as e:
    print(f"Import error: {e}. Please ensure you have all necessary dependencies installed.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 5}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None
