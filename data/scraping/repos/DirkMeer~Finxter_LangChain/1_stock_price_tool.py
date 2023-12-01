import yfinance as yf
from decouple import config
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool


chat_gpt_api = ChatOpenAI(
    model="gpt-3.5-turbo-0613",
    openai_api_key=config("OPENAI_API_KEY"),
    temperature=0,
)


class StockPriceTool(BaseTool):
    name: str = "yahoo_finance"
    description: str = "useful when you need to answer questions about the current stock price of a stock ticker"

    def _run(self, query: str) -> str:
        ticker = yf.Ticker(query)
        print(ticker)
        return f"{query} - ${ticker.info.get('currentPrice')}"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support asynchronous execution")


agent = initialize_agent(
    llm=chat_gpt_api,
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=[StockPriceTool()],
    verbose=True,
)


agent.run("Please tell me what a pineapple is?")
