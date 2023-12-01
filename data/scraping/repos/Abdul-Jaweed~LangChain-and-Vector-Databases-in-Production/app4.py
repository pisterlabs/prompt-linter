from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
deeplake_token = os.getenv("ACTIVELOOP_TOKEN")
os.environ["ACTIVELOOP_TOKEN"] = deeplake_token


llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create our documents 

texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

docs = text_splitter.create_documents(texts)


# create Deep Lake dataset

my_activeloop_org_id = "abduljaweed"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"

dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset

db.add_documents(docs)



retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)


tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Usefull for answering questions."
    ),
]


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


response = agent.run("When was Napoleone born?")
print(response)



# Load the existing Deep Lake dataset and specify the embedding function

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# Create the new documents 

texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 0
)

docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset

db.add_documents(docs)

# instantiate the wrapper class for GPT3

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

# Create a retriever from the db

retreival_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retrieval_qa=db.as_retriever()
)

# Instantiate a tool that uses the retriever

tool = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Usefull for answering questions."
    ),
]

# Create an agent that uses the tool

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Let's now test our agent with a new question.

response = agent.run("When was Michael Jordan born?")
print(response)

