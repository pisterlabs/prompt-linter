## DOESN'T WORK YET

from orchestrar import wp
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import TextLoader
from langchain.retrievers import SVMRetriever
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import logging
from dotenv import load_dotenv
from config import Config
import openai
import sys

# Load the .env file
load_dotenv()

# Configure OpenAI API key
cfg = Config()
try:
    openai.api_key = cfg.openai_api_key
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)

text = wp.get_all_articles('https://iminsweden.com/')

def analyse_text():
    # Your analysis logic goes here
    pass

def save_file(text):
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(text)

def load_file(filename='output.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def load_AI():
    loader = TextLoader("./output.txt")
    loader.load()

if __name__ == "__main__":
    #text = wp.get_all_articles('https://iminsweden.com/')
    #save_file(text)

    #loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    loader = TextLoader("output.txt")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    #print(all_splits)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    question = ("What is the Article Title of article with Article ID: 252?")
    docs = vectorstore.similarity_search(question)

    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
                                                      llm=ChatOpenAI(temperature=0))
    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    len(unique_docs)

    #llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    #qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    #result = qa_chain({"query": question})
    #print(result["result"])

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})
    print(result["result"])

# Save text to file
# save_file(text)

# Load text from file as a string
#loaded_text = load_file()

# You can now use 'loaded_text' as a string in your script
#print(loaded_text)