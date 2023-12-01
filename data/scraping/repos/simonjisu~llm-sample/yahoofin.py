import os
import yfinance as yf

from loguru import logger
from typing import List, Dict
from langchain.tools.base import ToolException

def get_ticker_info(ticker: str) -> str:
    """get general information of a ticker"""
    ticker_obj = yf.Ticker(ticker)
    s = ""
    for k, v in ticker_obj.info.items():
        s += f'{k}: {v}'
        s += '\n'
    return s

def get_income_stmt(ticker: str) -> str:
    """call yfinance to get income statement of a ticker"""
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.income_stmt
    return df.to_markdown()

def get_news(ticker: str) -> List[Dict]:
    """call yfinance to get news of a ticker"""
    ticker_obj = yf.Ticker(ticker)
    news = ticker_obj.news
    return news

def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "If there are several words related to user's query so that makes you confusing, please summrize the confusing part and return to user as final answer. Please give at least 3 examples of the confusing part."
    )