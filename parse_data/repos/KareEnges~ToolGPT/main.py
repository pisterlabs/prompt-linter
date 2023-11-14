from langchain import PromptTemplate, OpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, ConversationSummaryBufferMemory, \
    ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

import config

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, load_tools, StructuredChatAgent, AgentExecutor
from langchain.tools import Tool, StructuredTool, PythonREPLTool
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.utilities import GoogleSerperAPIWrapper as GSA
from langchain.chains import LLMChain, ConversationChain

# from tempfile import TemporaryDirectory


config.init()
# working_directory = TemporaryDirectory()
toolkit = FileManagementToolkit(
    # root_dir=str(working_directory.name)
    root_dir="."
)

Google = Tool(name="Google search",
              description="A tool to search recent infomation in Google",
              func=GSA().run)

"""
Google = DuckDuckGoSearchRun()
"""


def sumtool(end: int, start: int = 0, step: int = 1) -> int:
    """求和，从start到end以步长step累加，例如start=1,end=5,step=2,返回1+3+5的结果也就是9,start默认为0，step默认为1"""
    out = 0
    for i in range(start, end + 1, step):
        out += i
    return out


sumtool = StructuredTool.from_function(sumtool)

import GitTools
import CreateCode
#import memory

# llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY, model_name=config.model,
# streaming=config.streaming)
llm = OpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY, streaming=config.streaming)

tools = load_tools(["llm-math", "arxiv"], llm=llm)
tools += toolkit.get_tools()
tools += [sumtool, Google, CreateCode.CreateCodeTool]
tools += GitTools.gettools()
tools += [CreateCode.CreateCodeTool, PythonREPLTool()]


# memory = ConversationSummaryBufferMemory(llm=OpenAI(openai_api_key=config.OPENAI_API_KEY,
# streaming=config.streaming), max_token_limit=config.max_token_limit)

memory = ConversationBufferMemory(memory_key="chat_history")

def get_agent(tools):
    verbose = config.verbose

    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to 
    the following tools:"""
    suffix = """Begin!"
    
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = StructuredChatAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = StructuredChatAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    return agent


