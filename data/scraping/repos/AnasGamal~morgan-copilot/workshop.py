# %%
import requests
from pprint import pprint

#https://mm-gpt-chat-prod.azurewebsites.net
resp = requests.post("http://127.0.0.1:5000/chat", json={"matter_id": "12766106", "question": "Who is the plaintiff in this case?", "chat_history": ""}).json()
pprint(resp)

resp = requests.post("http://127.0.0.1:5000/chat", json={"matter_id": "12766106", "question": "What happened to him?", "chat_history": resp['updated_chat_history']}).json()
pprint(resp)

# %% [markdown]
# # Part 1: Data Gathering
# The first part of the vectorstore QA agent is creating the vector store. To do this, we scrape the documentation we want the agent to have access to and put it in a vector store. In this example, we used chroma, which allows you to create local vector stores, but for Morgan and Morgan we used a hosted version called Pinecone capable of hosting enormous amounts of data.

# %%
from bs4 import BeautifulSoup as Soup
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
import pickle

urls = ["https://docs.manim.community/en/stable/"]

# loader = RecursiveUrlLoader(url="https://docs.manim.community/en/stable/", max_depth=3, extractor=lambda x: Soup(x, "html.parser").text)
# docs = loader.load()

with open("manim_docs.pkl", "rb") as f:
    docs = pickle.load(f)

# %%
from pprint import pprint
print(f"{len(docs)} documents loaded")
pprint(docs[8].page_content)

# %%
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

db = Chroma.from_documents(split_docs, embedding=OpenAIEmbeddings())

# %% [markdown]
# # Part 2: QA Agent
# Now that we have our documentation in a vector store, we can use LangChain to create an agent capable of answering questions about these documents

# %%
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-4")

# %%
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

class DocsResponse(BaseModel):
    explanation: str = Field(description="The explanation answering the user's question")
    example: str = Field(description="A code example to support the explanation")

parser = PydanticOutputParser(pydantic_object=DocsResponse)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=chat)

prompt_template = """You are an AI assistant helping a user with a question about writing code with Manim. Answer the user's question using the documentation below and provide a code example to support your explanation.

FORMAT: {format_instructions}

DOCUMENTATION:
{context}

REQUEST: 
{question}

Answer the user's question according to the FORMAT above:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"], partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain_type_kwargs = {"prompt": PROMPT}

# %%
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4"), chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs, return_source_documents=True)

answer = qa_chain({"query": "How can I animate a circle in Manim?"})

parsed_answer = fixing_parser.parse(answer['result'])

print(parsed_answer.explanation)
pprint(parsed_answer.example)
print(answer['source_documents'])