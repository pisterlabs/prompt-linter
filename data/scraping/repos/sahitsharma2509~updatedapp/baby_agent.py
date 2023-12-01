


from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from typing import Optional
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from .babyagi_class import BabyAGI
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Pinecone
import pinecone


serp_api_key='3d856dbf3811b58136f49fd8ccfcac3c80bd3f22bacff9f9fec7f10fdcee11c3'
embeddings = OpenAIEmbeddings()
index_name = "test"
index = pinecone.Index(index_name=index_name)
vectorstore = Pinecone(
    index=pinecone.Index(index_name=index_name),
    embedding_function=embeddings.embed_query,
    text_key='text',
    namespace='baby_agi'
)




todo_prompt = PromptTemplate.from_template(
    "You are a task creation AI that creates new tasks with the following objective: {objective}"
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with tasks. Input: an objective to create a task list for. Output: a task list for that objective. Please be very clear what the objective is!",
    ),
]


prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# Logging of LLMChains
verbose = True
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
)


def get_baby_agi_response(objective: str):
    result = baby_agi({"objective": objective})
    return result