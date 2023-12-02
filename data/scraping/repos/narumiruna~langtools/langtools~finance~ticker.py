import pandas as pd
import yfinance as yf
from langchain.tools import BaseTool
from pydantic import BaseModel


class Ticker(BaseModel):
    Datetime: pd.Timestamp
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


class QueryTicker(BaseTool):
    name: str = 'query_ticker'
    description: str = ('A stock ticker tool. '
                        'Input should be a stock symbol. '
                        'The output will be the current stock price.')

    def _run(self, symbol: str) -> str:
        ticker: yf.Ticker = yf.Ticker(symbol)
        df = ticker.history(period='1d', interval='1m')
        df.reset_index(inplace=True)

        if df.empty:
            return f'No data found for symbol {symbol}'

        t = Ticker.parse_obj(df.iloc[-1].to_dict())
        return (f'Symbol: {symbol}\n'
                f'Open: {t.Open}\n'
                f'High: {t.High}\n'
                f'Low: {t.Low}\n'
                f'Close: {t.Close}\n'
                f'Time: {t.Datetime}\n')

    async def _arun(self, symbol: str) -> str:
        return self._run(symbol)
