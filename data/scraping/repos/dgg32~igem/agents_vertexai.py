#from langchain.tools import WikipediaQueryRun
#from langchain.utilities import WikipediaAPIWrapper
#import openai
from langchain.llms import VertexAI
import output_parser_vertexai
#from langchain.tools import DuckDuckGoSearchRun

from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.agents.tools import Tool


from langchain.chains import RetrievalQA

from langchain.embeddings.openai import OpenAIEmbeddings


from langchain.agents import ConversationalChatAgent
from langchain.agents import AgentExecutor

from langchain.memory import ConversationBufferMemory

import yaml

import os

from google.cloud import bigquery

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *


from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
#from langchain.llms.openai import OpenAI



with open("config.yaml", "r") as stream:
    try:
        PARAM = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#os.environ['OPENAI_API_KEY'] = PARAM["OPENAI_API_KEY"]
#openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = PARAM["GOOGLE_APPLICATION_CREDENTIALS_file"]

graph = Neo4jGraph(
    url=PARAM["neo4j_url"], username=PARAM["neo4j_username"], password=PARAM["neo4j_password"]
)

graph.refresh_schema()


vertex_ai = VertexAI(model_name="code-bison")

chain_neo4j = GraphCypherQAChain.from_llm(
    vertex_ai, graph=graph, verbose=True, return_direct=True
)


project = PARAM["bigquery_project_id"]

dataset = PARAM["bigquery_dataset_id"]

table = PARAM["bigquery_table_id"]


sqlalchemy_url = f"bigquery://{project}/{dataset}?credentials_path={os.environ['GOOGLE_APPLICATION_CREDENTIALS']}"

db = SQLDatabase.from_uri(sqlalchemy_url)

toolkit = SQLDatabaseToolkit(db=db, llm=vertex_ai)

bigquery_agent_executor = create_sql_agent(

    llm=vertex_ai,
    toolkit=toolkit,
    verbose=True,
    top_k=1000,

)

tools = [
    Tool(
        name="Neo4j_search",
        func=chain_neo4j.run,
        description=PARAM["neo4j_tool_description"],
    ),
    Tool(
        name="BigQuery_search",
        description=PARAM["BigQuery_description"],
        func=bigquery_agent_executor.run,
    )
]

agent_instructions = PARAM["agent_instruction"]


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=ChatOpenAI(model_name='gpt-4', temperature=0), tools=tools)

custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=VertexAI(temperature=0, max_output_tokens=500), tools=tools, output_parser=output_parser_vertexai.MyVertexOutputParser(),)


agent_executor = AgentExecutor.from_agent_and_tools(agent = custom_agent, tools=tools, memory=memory)
agent_executor.verbose = True

def ask_question(question):
    return agent_executor.run(question)