from decouple import config
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

from naturalchain.tools.calculator.tool import PythonCalculatorTool
from naturalchain.tools.graph.tool import UniswapGraphTool

OPENAI_API_KEY = config("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)  # type: ignore

if __name__ == "__main__":
    tools = [
        UniswapGraphTool(),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    question = f"Query the top 10 uniswap pools and print the address of the pool and the total value locked"
    #     question = f"Query the top 10 uniswap pools that have DAI as one of the tokens"
    #question = f"Give me the top 10 uniswap pools"
    response = agent.run(question)

    print(response)
