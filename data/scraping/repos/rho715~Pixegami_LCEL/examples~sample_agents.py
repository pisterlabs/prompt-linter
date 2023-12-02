from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models.openai import ChatOpenAI
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor

llm = ChatOpenAI()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# Set "SERPAPI_API_KEY" environment variable to your private key.
# https://serpapi.com/
search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        description="Search for information Google.",
        func=search.run,
    ),
    Tool(
        name="calculator",
        description="Use this tool to calculate the difference between numbers.",
        func=llm_math_chain.run,
    ),
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
agent_schema = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"]),
}

agent = agent_schema | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "What is the population difference between 🇺🇸 and 🇬🇧?"})