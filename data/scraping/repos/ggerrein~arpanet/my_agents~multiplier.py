from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.agents import initialize_agent, Tool


# This is an example of an arbitrary function that 
# can be used as a tool
def multiplier(a, b):
    return a * b

def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))

def get_tool():
    tool = Tool(
        name="Multiplier",
        func=parsing_multiplier,
        description="useful for when you need to multiply two numbers together. The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. For example, `1,2` would be the input if you wanted to multiply 1 by 2.",
    )
    return tool



