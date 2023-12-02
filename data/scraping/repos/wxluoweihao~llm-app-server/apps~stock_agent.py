from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
<<<<<<< HEAD
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '../')
=======
>>>>>>> main
from common.utils import Utils
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.agents import tool, initialize_agent, AgentType
import requests
from langchain.agents import AgentExecutor

<<<<<<< HEAD

@tool
def search_ticker_by_stockname(stock_name: str) -> int:
    """
    get the stock ticker using a stock or company name
    """
=======
@tool
def handle_failure() -> str:
    return "cannot help."

@tool
def search_ticker_by_stockname(stock_name: str) -> int:

>>>>>>> main
    print("executing search_tiker_by_stockname ...")
    url = "https://api.polygon.io/v3/reference/tickers?market=stocks&search={stock_name}&active=true&sort=name&apiKey=_O899h4QYZQiv8p_nB1bzp4xEs7sUGAV"\
        .format(stock_name=stock_name)
    response = requests.get(url)
    return response.json()

@tool
def get_stock_price_by_ticker(tiker_name: str, timespan: str, from_date:str, to_date:str) -> int:
<<<<<<< HEAD
    """
    finding open price, close price, mid prices for a ticker. 
    before that, use the tool search_ticker_by_stockname to get stock sticker using its name
    input requirements:
        * 2 input dates, which is a time range from yyyy-MM-dd to yyyy-MM-dd)
        * timespan(options are day, hour, week, month, year)
        * tiker name, need to call search_tiker_by_stockname to get tiker name for a stock
    """
=======
>>>>>>> main

    print("executing get_stick_price_by_tiker ...")
    url = "https://api.polygon.io/v2/aggs/ticker/{tiker_name}/range/1/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=120&apiKey=_O899h4QYZQiv8p_nB1bzp4xEs7sUGAV"\
        .format(tiker_name=tiker_name, timespan=timespan, from_date=from_date, to_date=to_date)
    response = requests.get(url)
    return response.json()

<<<<<<< HEAD
tools = [search_ticker_by_stockname, get_stock_price_by_ticker]

chatOpenAI = ChatOpenAI(openai_api_key = Utils.get_openai_key(), temperature=0.5, streaming=True)
# set_llm_cache(InMemoryCache())

#from langchain.tools.render import format_tool_to_openai_function
#llm_with_tools = chatOpenAI.bind(
#    functions=[format_tool_to_openai_function(t) for t in tools]
#)
=======
tools = [search_ticker_by_stockname, get_stock_price_by_ticker, handle_failure]

chatOpenAI = ChatOpenAI(openai_api_key = Utils.get_openai_key(), temperature=0)
# set_llm_cache(InMemoryCache())

from langchain.tools.render import format_tool_to_openai_function
llm_with_tools = chatOpenAI.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)
>>>>>>> main


agent_executor = initialize_agent(
    tools,
    chatOpenAI,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

<<<<<<< HEAD
agent_executor.run("what is the stock open price for amazon on 2023-09-28.")
=======
agent_executor.run("give me the close price for apple company at 2023-09-28.")
>>>>>>> main
