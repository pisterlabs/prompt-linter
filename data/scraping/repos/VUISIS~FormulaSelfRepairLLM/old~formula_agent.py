
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import json
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage
from langchain.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from formula_tools import LoadFormulaCode, QueryFormulaCode, DecodeFormulaCodeLLM
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from config import cfg
from prompts import FIX_CODE_PREFIX, SAMPLE_QUERY
from langchain.callbacks.base import BaseCallbackHandler

os.environ["OPENAI_API_KEY"] = cfg["OPENAI_API_KEY"]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "http://localhost:1984"

# class MyCustomHandler(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         # print(f"My custom handler, token: {token}")
#         pass


# llm = ChatOpenAI(temperature=0.0, streaming=True, callbacks=[MyCustomHandler()])
llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()



system_message = SystemMessage(content=FIX_CODE_PREFIX)
_prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


tools = [LoadFormulaCode, QueryFormulaCode, DecodeFormulaCodeLLM(llm=llm)]

# planner = load_chat_planner(llm)

# executor = load_agent_executor(llm, tools, verbose=True)
# agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory )


# suffix = """Begin!"

# {chat_history}
# Question: {input}
# {agent_scratchpad}"""

# prompt = ZeroShotAgent.create_prompt(
#     tools,
#     prefix=FIX_CODE_PREFIX,
#     suffix=suffix,
#     input_variables=["input", "chat_history", "agent_scratchpad"],
# )
# memory = ConversationBufferMemory(memory_key="chat_history")
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# _agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
# agent = AgentExecutor.from_agent_and_tools(
#     agent=_agent, tools=tools, verbose=True, memory=memory
# )

# agent.run(input=SAMPLE_QUERY)

agent = OpenAIFunctionsAgent(
    llm=llm,
    prompt=_prompt,
    tools=tools,
    memory=memory,
    verbose=True
    )

agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )

agent_executor.run(SAMPLE_QUERY)


# agent.run(FIX_CODE_PREFIX + sample_query)