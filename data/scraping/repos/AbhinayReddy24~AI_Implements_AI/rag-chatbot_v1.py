import openai
import os
import sys

from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import runnable
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from dotenv import load_dotenv

env = load_dotenv()
key = os.environ["OPENAI_API_KEY"]

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

# loader = PyPDFLoader("/Users/jessiedetchou/Downloads/Real-Time_Detection_of_DNS_Exfiltration_and_Tunneling_from_Enterprise_Networks-1.pdf", extract_images=True)
# pages = loader.load()

loader = PyPDFLoader(
    "documents/Real-Time_Detection_of_DNS_Exfiltration_and_Tunneling_from_Enterprise_Networks-1.pdf"
)
pages = loader.load()


# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(pages)

# Embed and store splits
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=key)
vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vector_store.as_retriever()

# create retrivealQA object
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

# Creates conversational ability and document retrieval capability
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=key),
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
)

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
