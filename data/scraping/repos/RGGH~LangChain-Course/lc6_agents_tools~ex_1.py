from langchain.agents import Tool
from langchain import LLMMathChain
from langchain.agents import initialize_agent
from langchain import OpenAI

llm = OpenAI()

llm_math_chain = LLMMathChain(llm=llm, verbose=True)
math_tool = Tool(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you need to answer questions about math.\
            This tool is only for math questions and nothing else. \
                Only input math expressions.",
    )

tools = []
# example of adding a tool retrospectively
tools.append(math_tool) 


zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
)

query_1 = "calculate the cube root of 625"
result = zero_shot_agent.run(query_1)

print(result)