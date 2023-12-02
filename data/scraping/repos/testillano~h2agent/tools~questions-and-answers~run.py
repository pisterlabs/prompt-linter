#######################################################
# QUESTIONS AND ANSWERS FROM DOCUMENTS - OpenAI-based #
#######################################################

# Imports
import os, sys, glob, pickle

# Script location
SCR_PATH = os.path.abspath(__file__)
SCR_DIR, SCR_BN = os.path.split(SCR_PATH)
REPO_DIR = os.path.abspath(SCR_DIR + "/../..")

# Chat history
CHAT_HISTORY = REPO_DIR + "/." + SCR_BN + "-chat-history"

# Basic checkings:
# Python3 version
major = sys.version_info.major
minor = sys.version_info.minor
micro = sys.version_info.micro
if major < 3 or (major == 3 and minor < 8):
  print("Python version must be >= 3.8.1 (current: {x}.{y}.{z}). Try alias it, i.e.: alias python3='/usr/bin/python3.9'".format(x=major, y=minor, z=micro))
  sys.exit(1)

# OpenAI version
try:
  apikey = os.environ["OPENAI_API_KEY"]
except:
  print("Please, export your OpenAI API KEY over 'OPENAI_API_KEY' environment variable")
  print("You may create the key here: https://platform.openai.com/account/api-keys")
  sys.exit(1)

# Load documents
print("Loading markdown documents under this directory ({}) ...".format(REPO_DIR))
wildcard=REPO_DIR + '/**/*.md'
markdowns = glob.glob(wildcard, recursive = True)
#print(markdowns)

from langchain.document_loaders import UnstructuredMarkdownLoader
loaders = [UnstructuredMarkdownLoader(os.path.join(SCR_DIR, md)) for md in markdowns]
#print("Loading URL sources ...")
#from langchain.document_loaders import UnstructuredURLLoader
#loaders.append(UnstructuredURLLoader(["https://prezi.com/p/1ijxuu0rt-sj/?present=1)"]))
documents = []
for loader in loaders:
  documents.extend(loader.load())

# create index
#from langchain.indexes import VectorstoreIndexCreator
#index = VectorstoreIndexCreator().from_loaders(loaders)
#index.query_with_sources("here the query...")

# Indexing data:
print("Indexing data ...")

# Split the document into chunks:
from langchain.text_splitter import MarkdownTextSplitter
text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Select which enbeddings we want to use
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Create the vectorstore to use as the index
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(texts, embeddings)

# Expose this index in a retriever interface
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":9})

# Create a chain to answer questions
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever) # , return_source_documents=True)
#vectordbkwargs = {"search_distance": 0.9}

# Chat history: read if exists
chat_history = []
try:
  with open(CHAT_HISTORY, 'rb') as f: # open in binary mode
    chat_history = pickle.load(f) # Deserialize the array from the file
except:
  pass

# Main logic
while True:
  query = input("\nAsk me anything (0 = quit): ")
  if query == "0": break
  result = qa({"question": query, "chat_history": chat_history}) # , "vectordbkwargs": vectordbkwargs1})
  answer = result["answer"]
  print(answer)
  chat_history.append([query, answer])

print("\n[saving chat history ...]\n")
with open(CHAT_HISTORY, 'wb') as f: # write in binary mode
  pickle.dump(chat_history, f) # serialize the array and write it to the file

print("Done !")
