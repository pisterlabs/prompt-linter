from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from dotenv import load_dotenv

from pretty_print_callback_handler import PrettyPrintCallbackHandler


load_dotenv()
my_callback = PrettyPrintCallbackHandler()


search = SerpAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI(temperature=0)
llm.callbacks = [my_callback]

#    tools,
agent_chain = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)
agent_chain.callbacks = [my_callback]

# agent_chain.run(input="hi, i am bob")

# agent_chain.run(input="what's my name?")
agent_chain.run(input="I have insulin resistence and I'm sensitive to sugar.")

agent_chain.run("what are some good dinners to make this week, if i like thai food?")

# agent_chain.run(input="whats the current temperature in pomfret?")
