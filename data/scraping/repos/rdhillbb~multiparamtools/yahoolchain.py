# Example of multi-parameter tools with instance Data
from typing import List
from datetime import datetime, timedelta
import yfinance as yf
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory
import nltk
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.tools import BaseTool, StructuredTool, Tool, tool

# Create LLM.
llm2 = ChatOpenAI(
    temperature=0,
    streaming=False,
    # callbacks=[StreamingStdOutCallbackHandler()],
    callbacks=[
        FinalStreamingStdOutCallbackHandler(
            answer_prefix_tokens=["Final", " Answer", ":"]
        )
    ],
    model="gpt-4-0613",
)
memory = ConversationBufferMemory(memory_key="chat_history")
# Instance Data Objects
# Parameter Objects Definition
class FuncP1STR:
    def __init__(self, uid, req, run_function):
        self.uid = uid
        self.req = req
        self.run_function = run_function

    def run(self, param1):
        return self.run_function(param1, self.uid, self.req)


# 4 String Parameter Function
class FuncP4ASTR:
    def __init__(self, uid, req, run_function):
        self.uid = uid
        self.req = req
        self.run_function = run_function

    def run(self, param1: str, param2:str,param3:str,param4:str ):
        return self.run_function(pram1, pram2, param3, param4, self.uid, self.req)


# 2 String Parameter Function
class FuncP2STR:
    def __init__(self, uid, req, run_function):
        self.uid = uid
        self.req = req
        self.run_function = run_function

    def run(self, param1: str, param2: str):
        return self.run_function(param1, param2, self.uid, self.req)


# 2 String Parameter 1 Int
class FuncP1STR2INT:
    def __init__(self, uid, req, run_function):
        self.uid = uid
        self.req = req
        self.run_function = run_function

    def run(self, param1: str, param2: str):
        return self.run_function(param1, param2, num, self.uid, self.req)


# Tools/Functions

# Parameter Opjects Definition
class StockPriceYTool:
    """Input for Stock price check."""

    def __init__(self, uid, req, run_function):
        self.uid = uid
        self.req = req
        self.run_function = run_function

    def run(self, symbol):
        return self.run_function(symbol, self.uid, self.req)


def get_stock_price(symbol, uid, req):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period="1d")
    return round(todays_data["Close"][0], 2)


priceChangeName = "get_price_change_percent"
priceChangeDescription = "Useful for when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"


class GetPriceChangePercent:
    def __init__(self, uid, req, run_function):
        self.uid = uid
        self.req = req
        self.run_function = run_function

    def run(self, symbol: str, days_ago: int):
        return self.run_function(symbol, days_ago, self.uid, self.req)


def get_price_change_percent(symbol, days_ago, uid, req):
    """Input for Stock ticker check. for percentage check"""
    ticker = yf.Ticker(symbol)

    # Get today's date
    end_date = datetime.now()

    # Get the date N days ago
    start_date = end_date - timedelta(days=days_ago)

    # Convert dates to string format that yfinance can accept
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    # Get the historical data
    historical_data = ticker.history(start=start_date, end=end_date)

    # Get the closing price N days ago and today's closing price
    old_price = historical_data["Close"].iloc[0]
    new_price = historical_data["Close"].iloc[-1]

    # Calculate the percentage change
    percent_change = ((new_price - old_price) / old_price) * 100

    return round(percent_change, 2)


# Best Performing Stock
bestPerformingName = "get_best_performing"
bestPerformingDescription = "Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"


class GetBestPerformingStock:
    """Input for Stock ticker check. for percentage check"""

    def __init__(self, uid, req, run_function):
        self.uid = uid
        self.req = req
        self.run_function = run_function

    def run(self, stocktickers: List[str], days_ago: int):
        return self.run_function(stocktickers, days_ago, self.uid, self.req)


def calculate_performance(symbol, days_ago):
    ticker = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    historical_data = ticker.history(start=start_date, end=end_date)
    old_price = historical_data["Close"].iloc[0]
    new_price = historical_data["Close"].iloc[-1]
    percent_change = ((new_price - old_price) / old_price) * 100
    return round(percent_change, 2)


def get_best_performing(stocks, days_ago, uid, req):
    best_stock = None
    best_performance = None
    for stock in stocks:
        try:
            performance = calculate_performance(stock, days_ago)
            if best_performance is None or performance > best_performance:
                best_stock = stock
                best_performance = performance
        except Exception as e:
            print(f"Could not calculate performance for {stock}: {e}")
    return best_stock, best_performance


# Stock News


class StockNewsYTool:
    """Input for Stock price check."""

    def __init__(self, uid, req, run_function):
        self.uid = uid
        self.req = req
        self.run_function = run_function

    def run(self, symbol):
        return self.run_function(symbol, self.uid, self.req)


class Article:
    def __init__(self, title, publisher, link, time_published, related_tickers):
        self.title = title
        self.publisher = publisher
        self.link = link
        self.time_published = time_published
        self.related_tickers = related_tickers


def process_json_data(json_data):
    processed_data = []

    for item in json_data:
        title = item["title"]
        publisher = item["publisher"]
        link = item["link"]

        # Convert Unix timestamp to human-readable format
        timestamp = item["providerPublishTime"]
        datetime_object = datetime.fromtimestamp(timestamp)
        formatted_time = datetime_object.strftime("%Y-%m-%d %H:%M")

        related_tickers = item["relatedTickers"]

        article = Article(title, publisher, link, formatted_time, related_tickers)
        processed_data.append(article)

    return processed_data


nameGetStockNews = "get_stock_news"
descriptionGetStockNews = "market news and insights on stocks and indices, providing comprehensive coverage and global perspective for informed decision-making."


def get_stock_news(ticker, uid, req):
    stock = yf.Ticker(ticker)
    news_array = process_json_data(stock.news)
    long_string = ""

    for article in news_array:
        long_string += "Title: " + article.title + "\n"
        long_string += "Publisher: " + article.publisher + "\n"
        long_string += "Link: " + article.link + "\n"
        long_string += "Time Published: " + article.time_published + "\n"
        long_string += "Related Tickers: " + ", ".join(article.related_tickers) + "\n"
        long_string += "\n"
    return long_string


tools = []
tools.append(
    StructuredTool.from_function(
        StockNewsYTool(uid="c290-ww3-123cd", req="sync", run_function=get_stock_news).run,
        name=nameGetStockNews,
        description=descriptionGetStockNews,
    )
)
tools.append(
    StructuredTool.from_function(
        StockPriceYTool(uid="c290-ww3-123cd", req="sync", run_function=get_stock_price).run,
        name="get_stock_ticker_price",
        description="Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API",
    )
)
tools.append(
    StructuredTool.from_function(
        GetPriceChangePercent(
            uid="c290-ww3-123cd", req="sync", run_function=get_price_change_percent
        ).run,
        name=priceChangeName,
        description=priceChangeDescription,
    )
)

tools.append(
    StructuredTool.from_function(
        GetBestPerformingStock(
            uid="c290-ww3-123cd", req="sync", run_function=get_best_performing
        ).run,
        name=bestPerformingName,
        description=bestPerformingDescription,
    )
)


agent = initialize_agent(
    tools,
    llm2,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)
print(
    agent.run(
        input="I want the stock price for each of the following stocks;Tesla, Microsoft. Out of these stocks which one had the best peformance over the past 60 days? I would like the news on each stock as well."
    )
)
print(
    agent.run(
        input="I want the stock price for each of the following stocks;Tesla, Microsoft. Out of these stocks which one had the best peformance over the past 10 days? I would like the news on each stock as well."
    )
)
