import openai

from langchain.utilities import SerpAPIWrapper

from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.agents.tools import Tool

from langchain.vectorstores import Qdrant

from langchain.chains import RetrievalQA

from langchain.document_loaders.csv_loader import CSVLoader


from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Qdrant

from langchain.agents import ConversationalChatAgent
from langchain.agents import AgentExecutor

from langchain.memory import ConversationBufferMemory

import yaml

import os

with open("config.yaml", "r") as stream:
    try:
        PARAM = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

os.environ['OPENAI_API_KEY'] = PARAM["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

graph = Neo4jGraph(
    url="bolt://localhost:7687", username=PARAM["neo4j_username"], password=PARAM["neo4j_password"]
)

graph.refresh_schema()

os.environ['OPENAI_API_KEY'] = PARAM["OPENAI_API_KEY"]

chain_neo4j = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, return_direct=True
)

os.environ["SERPAPI_API_KEY"] = PARAM["serpapi_key"]

# params = {
#     "engine": "wikipedia",
#     "gl": "jp",
#     "hl": "jp",
# }

serpai = SerpAPIWrapper()

loader = CSVLoader("for_neo4j/node_trial.tsv", source_column="NCT", csv_args={
    'delimiter': '\t'
})
docs = loader.load()

url = PARAM["qdrant_URL"]
api_key = PARAM["qdrant_API_KEY"]

embeddings = OpenAIEmbeddings()

qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url = url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="my_documents",
)

retriever = qdrant.as_retriever()

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name='gpt-4', temperature=0), chain_type="stuff", retriever=retriever)

tools = [
    Tool(
        name="Neo4j_search",
        func=chain_neo4j.run,
        description="useful for when you need to answer questions about clinical trials. This is your default tool. Input should be in the form of a question containing full context",
    ),
    Tool(
        name="Vector_search",
        func=qa.run,
        description="Utilize this tool when the user asks for similarity search and when he explicitly asks you to use Vector_search. Input should be in the form of a question containing full context",
    ),
    Tool(
        name="Serp_search",
        description="Utilize this tool when the user asks for information of a company or an institution and when he explicitly asks you to search the web. Input should be in the form of a question containing full context",
        func=serpai.run,
    )
]

agent_instructions = "Try 'Neo4j_search' tool first. If Neo4j returns good result, return it and exit the chain. Use the other tools only if the user explicitly requests you to."


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=ChatOpenAI(model_name='gpt-4', temperature=0), tools=tools)



agent_executor = AgentExecutor.from_agent_and_tools(agent = custom_agent, tools=tools, memory=memory)
agent_executor.verbose = True

def ask_question(question):
    return agent_executor.run(question)