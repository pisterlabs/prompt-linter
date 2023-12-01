# https://python.langchain.com/docs/use_cases/question_answering/how_to/conversational_retrieval_agents

# Let's try to create our first agent using all the knowledge we have acquired
# so far.

import os
import openai
import pandas as pd

from dotenv import load_dotenv
from pprint import pprint

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage
)

load_dotenv()  # This will load the variables from .env file

OPENAI_KEY = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-ada-002"

# As we have seen in the previous section, everything start with a system prompt
# that we will use to initialize our Agent. Let's create a simple one for our
# movies recommender system.
system_message = SystemMessage(
    content=(
        "Your name is Hugo and you are a movie expert."
        "You are helping people to find the best movies to watch on Netflix."
    )
)

# Let's create our first tool for Hugo to be able to search over the available
# movies
embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_KEY, model=EMBEDDING_MODEL, client=openai.Embedding
)

db = Chroma(persist_directory="data/chroma_db",
            embedding_function=embedding, collection_name="movies")
retriever = db.as_retriever()

tool = create_retriever_tool(
    retriever,
    "netflix_movies_retriever",
    "Netflix movies Retriever",
)
tools = [tool]

llm = ChatOpenAI(temperature=0)
agent_executor = create_conversational_retrieval_agent(
    llm, tools, system_message=system_message, verbose=True)

result = agent_executor(
    {"input": "Hi Hugo! I would like to watch a comedy with a cat and a dog, could you please suggest me some movies?"})

pprint(result)
print()
print(">>> LET'S TALK ABOUT THE ERROR AND HOW TO FIX IT <<<")