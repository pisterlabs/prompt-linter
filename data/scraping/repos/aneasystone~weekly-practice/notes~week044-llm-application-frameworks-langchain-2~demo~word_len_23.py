from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field

# llm
llm = ChatOpenAI(temperature=0, verbose=True)

# tools
def get_word_length(word: str, excluding_hyphen: bool) -> int:
    if excluding_hyphen:
        return len(word.replace('-', ''))
    else:
        return len(word)

class WordLengthSchema(BaseModel):
    word: str = Field(description = "the word to be calculating")
    excluding_hyphen: bool = Field(description = "excluding the hyphen or not, default to false")

tools = [
    StructuredTool.from_function(
        func=get_word_length,
        name="get_word_length",
        description="Returns the length of a word.",
        args_schema=WordLengthSchema
    )
]

# prompt
system_message = SystemMessage(
    content="You are very powerful assistant, but bad at calculating lengths of words."
)
prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)

# create an agent
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

# create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# run the agent executor
result = agent_executor.run("how many letters in the word 'weekly-practice', including the hyphen?")
print(result)
result = agent_executor.run("how many letters in the word 'weekly-practice', excluding the hyphen?")
print(result)
