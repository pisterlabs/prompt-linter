# Imports
import os
from dotenv import load_dotenv
import textwrap
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Set the maximum line width
max_line_width = 80

# Load environment variables from .env file
load_dotenv()

embeddings = OpenAIEmbeddings()

# Initialize OpenAI wrapper
llm = OpenAI(temperature=0.9)

embeddings = OpenAIEmbeddings()

root_dir = './clone-nayms'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

db = DeepLake.from_documents(texts, embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 20

model = ChatOpenAI(temperature=0)
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


def ask_questions(qa, questions, chat_history=None, max_line_width=80):
    if chat_history is None:
        chat_history = []
    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        wrapped_answer = textwrap.fill(result['answer'], width=max_line_width)
        print(f"-> **Question**: {question}\n")
        print(f"**Answer**:\n{wrapped_answer}\n")


questions = [
    "Give me a list of all of the methods in AdminFacet.",
]
