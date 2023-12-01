from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from dotenv import load_dotenv
import os
import openai

from llama_index import StorageContext, load_index_from_storage
from pathlib import Path
from llama_index.indices.base import BaseIndex

from langchain.agents import load_tools
from pydantic import BaseModel, Field

from inspect import signature

import uuid
import sys


import logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

from utils.custom_agent import CustomTool, Instruction, SystemOutput, HumanMessage, ChatHistory, DeterministicPlannerAndExecutor


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("OPENAI_API_KEY is not set")
else:
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


PWD = Path(__file__).parent
DATA_PATH = Path.joinpath(PWD, "../data")

ACCOUNT_PATH = Path.joinpath(DATA_PATH, "accounts")
ARTICLE_PATH = Path.joinpath(DATA_PATH, "articles")

ACCOUNT_INDEX_PATH = Path.joinpath(DATA_PATH, "account_index")
ARTICLE_INDEX_PATH = Path.joinpath(DATA_PATH, "article_index")

if not ACCOUNT_INDEX_PATH.exists():
    ACCOUNT_INDEX_PATH.mkdir()

if not ARTICLE_INDEX_PATH.exists():
    ARTICLE_INDEX_PATH.mkdir()



if __name__ == "__main__":

    class IndexInput(BaseModel):
        query: str = Field(description="A semantic query to execute against the index.")
        attributes_to_find: str = Field(description="A comma separated list of attributes to find in the document.")
        number_of_results: int = Field(description="The number of results to return in a single string. Greater than 0")

    class IndexOutput(BaseModel):
        result: str = Field(description="The result of the query.")
        

    account_storage_context = StorageContext.from_defaults(persist_dir=ACCOUNT_INDEX_PATH)
    account_index = load_index_from_storage(account_storage_context)
    
    article_storage_context = StorageContext.from_defaults(persist_dir=ARTICLE_INDEX_PATH)
    article_index = load_index_from_storage(article_storage_context)

    def account_query(input: IndexInput) -> IndexOutput:

        new_query = f"Find the attributes: {input.attributes_to_find} of the account with the following query: {input.query}"

        return IndexOutput(
            result=str(account_index.as_query_engine(similarity_top_k=input.number_of_results).query(new_query))
        )
    
    def article_query(input: IndexInput) -> IndexOutput:

        new_query = f"Find the attributes: {input.attributes_to_find} of the article with the following query: {input.query}"

        return IndexOutput(
            result=str(article_index.as_query_engine(similarity_top_k=input.number_of_results).query(new_query))
        )

    tools = [
        CustomTool(
            function=account_query,
            description="This does an embedding search on the accounts in the database and returns exactly the information you requested. The function returns a summary of exactly what you asked for. Specify the exact information you want returned with any attributes.",
        ),

        CustomTool(
            function=article_query,
            description="This does an embedding search on the articles in the database and returns exactly the information you requested. The function returns a summary of exactly what you asked for. Specify the exact information you want returned with any attributes.",
        ),
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-4")


    history = ChatHistory()
    agent = DeterministicPlannerAndExecutor(tools=tools, llm=llm, history=history)

    while True:
        query = input("Enter a query: ")

        result = agent.predict(query=query)

        print('-'*100)
        print(result)
        print('-'*100)




# def main():

#     account_storage_context = StorageContext.from_defaults(persist_dir=ACCOUNT_INDEX_PATH)
#     account_index = load_index_from_storage(account_storage_context)
    
#     article_storage_context = StorageContext.from_defaults(persist_dir=ARTICLE_INDEX_PATH)
#     article_index = load_index_from_storage(article_storage_context)


#     tools = [
#         Tool(
#             name="account_index",
#             func=lambda q: str(account_index.as_query_engine().query(q)),
#             description="This queries the accounts of users in the database with a semantic query, and returns one or more accounts as a single string, information includes: name, email, phone number, address, and other information. Specify which information you want returned.",
#             return_direct=True,
#         ),

#         Tool(
#             name="article_index",
#             func=lambda q: str(article_index.as_query_engine().query(q)),
#             description="This queries the articles in the database with a semantic query, and returns one or more articles as a single string, information includes: title, content, author, date, and url and source. Specify which information you want returned.",
#             return_direct=True,
#         ),
#     ]

#     memory = ConversationBufferMemory(memory_key="chat_history")
#     llm = ChatOpenAI(temperature=0.5, model="gpt-4")
    
#     agent_executor = initialize_agent(
#         tools, llm, memory=memory
#     )

#     x = agent_executor.run(input="Find an article about black holes. What was the Source?")
#     print(x)


# if __name__ == "__main__":
#     main()