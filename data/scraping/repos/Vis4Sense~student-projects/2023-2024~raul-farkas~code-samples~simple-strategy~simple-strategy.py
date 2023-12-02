#!/usr/bin/env python3
import asyncio
from datetime import datetime
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from alpaca.data.requests import StockBarsRequest
import pandas as pd
from alpaca.data.requests import StockBarsRequest, TimeFrame
from openai import AsyncOpenAI
import plotly.graph_objects as go

pd.options.plotting.backend = "plotly"


def get_news(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Given a symbol and a timeframe, return the news about that symbol"""
    key = os.environ["ALPACA_KEY"]
    secret = os.environ["ALPACA_SECRET"]
    client = NewsClient(api_key=key, secret_key=secret)

    request_params = NewsRequest(symbols=symbol, start=start, end=end, limit=50)
    news = [x.model_dump() for x in client.get_news(request_params).news]
    return pd.DataFrame.from_records(news, columns=news[0].keys())


async def analyse_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse the sentiment of the news headlines"""
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API"],
    )
    df["sentiment"] = None

    async def get_sentiment(index, row, df, retry=3) -> None:
        for _ in range(retry):
            try:
                chat_completion = await client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "Instruction: What is the sentiment of this news for the company AAPL? Please choose an answer from (negative/neutral/positive). Respond with one word.\\nInput: {}\\nAnswer:".format(
                                row["headline"]
                            ),
                        }
                    ],
                    model="gpt-3.5-turbo",
                    timeout=10,
                )
                break
            except:
                pass
        # Set the sentiment
        df.at[index, "sentiment"] = chat_completion.choices[0].message.content.lower()

    waiting_tasks = []
    for index, row in df.iterrows():
        task = asyncio.create_task(get_sentiment(index, row, df))
        waiting_tasks.append(task)
    await asyncio.gather(*waiting_tasks)
    return df


def map_news_to_stock_bars(news: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Map the news to the stock bars. This will create a new column in the bars dataframe"""
    news["timestamp"] = pd.to_datetime(news["created_at"])
    # Aproximate minutes to nearest minute
    news["timestamp"] = news["timestamp"].dt.floor("min")
    # Remove duplicates
    news = news.drop_duplicates(subset=["timestamp"])
    # Match news to the bars based on timestamp index
    news = news.set_index("timestamp")
    bars = bars.join(news, how="left")

    return bars


def get_stock_bars(symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Get stock bars from Alpaca.markets"""
    key = os.environ["ALPACA_KEY"]
    secret = os.environ["ALPACA_SECRET"]
    client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    request_params = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Minute, start=start, end=end
    )

    bars = client.get_stock_bars(request_params)

    return bars.df


def add_moving_averages(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Given a df and periods add moving averages of the close."""
    for period in periods:
        df[f"{period}MA"] = df["close"].rolling(period).mean()

    return df


def add_trading_signals(df: pd.DataFrame):
    """
    Given the market data, article sentiment and moving averages generate trading signals with the following logic:
    Using a 20 minute window:
        If the 10 minute moving average is below the 80 minute moving average and the sentiment is negative, buy.
        If the 10 minute moving average is above the 80 minute moving average and the sentiment is positive, sell.
        Buy or sell only if the previous signal was not a buy or sell.
    """
    df.loc[:, "signal"] = "hold"

    # Move over the dataframe in 20 minute windows
    for i in range(0, len(df), 20):
        # Get previous signal
        previous_signal = df.iloc[i - 1]["signal"]

        # Get the 5 minute window
        window = df.iloc[i : i + 20]
        # Get the 5 minute moving average
        ma5 = window["10MA"].mean()
        # Get the 80 minute moving average
        ma80 = window["80MA"].mean()

        # Get a list of all sentiments in the window
        sentiments = [x for x in window["sentiment"].tolist() if type(x) is str]
        # Get the most common sentiment in the window
        sentiment = None
        if sentiments:
            sentiment = max(set(sentiments), key=sentiments.count)

        if ma5 > ma80 and sentiment == "positive" and previous_signal != "buy":
            df.at[window.index[0], "signal"] = "buy"
        elif ma5 < ma80 and sentiment == "negative" and previous_signal != "sell":
            df.at[window.index[0], "signal"] = "sell"

    return df


def plot_trading(df: pd.DataFrame):
    """Plot the trading strategy."""

    fig = go.Figure()
    # Add scatter plots for sentiment
    fig.add_trace(
        go.Scatter(
            x=df[df["sentiment"] == "positive"].index.get_level_values("timestamp"),
            y=df[df["sentiment"] == "positive"]["close"],
            mode="markers",
            name="Positive",
            marker=dict(color="green", size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[df["sentiment"] == "neutral"].index.get_level_values("timestamp"),
            y=df[df["sentiment"] == "neutral"]["close"],
            mode="markers",
            name="Neutral",
            marker=dict(color="purple", size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[df["sentiment"] == "negative"].index.get_level_values("timestamp"),
            y=df[df["sentiment"] == "negative"]["close"],
            mode="markers",
            name="Negative",
            marker=dict(color="red", size=10),
        )
    )

    # Add scatter plots for buy and sell signals
    fig.add_trace(
        go.Scatter(
            x=df[df["signal"] == "buy"].index.get_level_values("timestamp"),
            y=df[df["signal"] == "buy"]["close"],
            mode="markers",
            name="Buy",
            marker=dict(color="green", symbol="triangle-up", size=12),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[df["signal"] == "sell"].index.get_level_values("timestamp"),
            y=df[df["signal"] == "sell"]["close"],
            mode="markers",
            name="Sell",
            marker=dict(color="red", symbol="triangle-up", size=12),
        )
    )

    # Add line plots for close, 10MA, and 80MA
    fig.add_trace(
        go.Scatter(
            x=df.index.get_level_values("timestamp"),
            y=df["close"],
            mode="lines",
            name="Close",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index.get_level_values("timestamp"),
            y=df["10MA"],
            mode="lines",
            name="10MA",
            line=dict(color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index.get_level_values("timestamp"),
            y=df["80MA"],
            mode="lines",
            name="80MA",
            line=dict(color="orange"),
        )
    )

    # Set layout properties
    fig.update_layout(
        title="Stock Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        autosize=False,
        width=1200,
        height=600,
    )

    # Show the figure
    fig.show()


async def main():
    df = get_stock_bars(["AAPL"], datetime(2023, 11, 7), datetime(2023, 11, 8))
    df_news = get_news("AAPL", datetime(2023, 11, 7), datetime(2023, 11, 8))

    df_news = await analyse_sentiment(df_news)

    # Get all the rows where the index symbol is AAPL
    aapl = df[df.index.get_level_values("symbol") == "AAPL"]
    aapl = add_moving_averages(aapl, [10, 80])
    aapl = map_news_to_stock_bars(df_news, aapl)
    aapl = add_trading_signals(aapl)

    plot_trading(aapl)


if __name__ == "__main__":
    asyncio.run(main())
