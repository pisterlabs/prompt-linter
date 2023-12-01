# bez agenta
import json
from langchain.schema import HumanMessage, AIMessage, ChatMessage, FunctionMessage
from langchain.tools import format_tool_to_openai_function
from langchain.agents import Tool
# sa agentom
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from typing import Optional, Type, List
from langchain.tools import BaseTool
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import os
from mojafunkcja import st_style, init_cond_llm

st_style()
# funkcije za agenta


def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return round(todays_data['Close'][0], 2)


def get_price_change_percent(symbol, days_ago):
    ticker = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    historical_data = ticker.history(start=start_date, end=end_date)
    # Get the closing price N days ago and today's closing price
    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]
    # Calculate the percentage change
    percent_change = ((new_price - old_price) / old_price) * 100

    return round(percent_change, 2)


def calculate_performance(symbol, days_ago):
    ticker = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    historical_data = ticker.history(start=start_date, end=end_date)
    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]
    percent_change = ((new_price - old_price) / old_price) * 100
    return round(percent_change, 2)


def get_best_performing(stocks, days_ago):
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

# templatei za model i toolove za agenta


class StockPriceCheckInput(BaseModel):
    """Input for Stock price check."""

    stockticker: str = Field(...,
                             description="Ticker symbol for stock or index")


class StockPriceTool(BaseTool):
    name = "get_stock_ticker_price"
    description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):
        # print("i'm running")
        price_response = get_stock_price(stockticker)

        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput


class StockChangePercentageCheckInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stockticker: str = Field(...,
                             description="Ticker symbol for stock or index")
    days_ago: int = Field(..., description="Int number of days to look back")


class StockPercentageChangeTool(BaseTool):
    name = "get_price_change_percent"
    description = "Useful for when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stockticker: str, days_ago: int):
        price_change_response = get_price_change_percent(stockticker, days_ago)

        return price_change_response

    def _arun(self, stockticker: str, days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockChangePercentageCheckInput


class StockBestPerformingInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stocktickers: List[str] = Field(...,
                                    description="Ticker symbols for stocks or indices")
    days_ago: int = Field(..., description="Int number of days to look back")


class StockGetBestPerformingTool(BaseTool):
    name = "get_best_performing"
    description = "Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stocktickers: List[str], days_ago: int):
        price_change_response = get_best_performing(stocktickers, days_ago)

        return price_change_response

    def _arun(self, stockticker: List[str], days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput


def main():

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    tools = [StockPriceTool(), StockPercentageChangeTool(),
             StockGetBestPerformingTool()]

    # input

    st.subheader("Ask a question about stocks")
    st.caption("Example od using the agent with OpenAI functions")
    model, temp = init_cond_llm()
    chat_model = ChatOpenAI(model=model, temperature=temp)

    col1, col2 = st.columns(2)
    with col1:

        with st.form(key='my_form', clear_on_submit=True):
            input_message = st.text_input("Enter your question: ")
            submit_button = st.form_submit_button(label='Submit')
            if submit_button and input_message:
                open_ai_agent = initialize_agent(tools,
                                                 chat_model,
                                                 agent=AgentType.OPENAI_FUNCTIONS,
                                                 verbose=False)
                agent_answer = open_ai_agent.run(input_message)
                st.write(agent_answer)

    with col2:
        with st.expander("Show code"):
            st.code("""# bez agenta
                        import json
                        from langchain.schema import HumanMessage, AIMessage, ChatMessage, FunctionMessage
                        from langchain.tools import format_tool_to_openai_function
                        from langchain.agents import Tool
                        # sa agentom
                        from pydantic import BaseModel, Field
                        from langchain.chat_models import ChatOpenAI
                        from langchain.agents import AgentType
                        from langchain.agents import initialize_agent
                        from typing import Optional, Type, List
                        from langchain.tools import BaseTool
                        import yfinance as yf
                        from datetime import datetime, timedelta
                        import streamlit as st
                        import os
                        from mojafunkcja import st_style

                        st_style()
                        # funkcije za agenta


                        def get_stock_price(symbol):
                            ticker = yf.Ticker(symbol)
                            todays_data = ticker.history(period='1d')
                            return round(todays_data['Close'][0], 2)


                        def get_price_change_percent(symbol, days_ago):
                            ticker = yf.Ticker(symbol)
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=days_ago)
                            start_date = start_date.strftime('%Y-%m-%d')
                            end_date = end_date.strftime('%Y-%m-%d')
                            historical_data = ticker.history(start=start_date, end=end_date)
                            # Get the closing price N days ago and today's closing price
                            old_price = historical_data['Close'].iloc[0]
                            new_price = historical_data['Close'].iloc[-1]
                            # Calculate the percentage change
                            percent_change = ((new_price - old_price) / old_price) * 100

                            return round(percent_change, 2)


                        def calculate_performance(symbol, days_ago):
                            ticker = yf.Ticker(symbol)
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=days_ago)
                            start_date = start_date.strftime('%Y-%m-%d')
                            end_date = end_date.strftime('%Y-%m-%d')
                            historical_data = ticker.history(start=start_date, end=end_date)
                            old_price = historical_data['Close'].iloc[0]
                            new_price = historical_data['Close'].iloc[-1]
                            percent_change = ((new_price - old_price) / old_price) * 100
                            return round(percent_change, 2)


                        def get_best_performing(stocks, days_ago):
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

                        # templatei za model i toolove za agenta


                        class StockPriceCheckInput(BaseModel):
                            ""Input for Stock price check.""

                            stockticker: str = Field(...,
                                                    description="Ticker symbol for stock or index")


                        class StockPriceTool(BaseTool):
                            name = "get_stock_ticker_price"
                            description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"

                            def _run(self, stockticker: str):
                                # print("i'm running")
                                price_response = get_stock_price(stockticker)

                                return price_response

                            def _arun(self, stockticker: str):
                                raise NotImplementedError("This tool does not support async")

                            args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput


                        class StockChangePercentageCheckInput(BaseModel):
                            ""Input for Stock ticker check. for percentage check""

                            stockticker: str = Field(...,
                                                    description="Ticker symbol for stock or index")
                            days_ago: int = Field(..., description="Int number of days to look back")


                        class StockPercentageChangeTool(BaseTool):
                            name = "get_price_change_percent"
                            description = "Useful for when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"

                            def _run(self, stockticker: str, days_ago: int):
                                price_change_response = get_price_change_percent(stockticker, days_ago)

                                return price_change_response

                            def _arun(self, stockticker: str, days_ago: int):
                                raise NotImplementedError("This tool does not support async")

                            args_schema: Optional[Type[BaseModel]] = StockChangePercentageCheckInput


                        class StockBestPerformingInput(BaseModel):
                            ""Input for Stock ticker check. for percentage check""

                            stocktickers: List[str] = Field(...,
                                                            description="Ticker symbols for stocks or indices")
                            days_ago: int = Field(..., description="Int number of days to look back")


                        class StockGetBestPerformingTool(BaseTool):
                            name = "get_best_performing"
                            description = "Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"

                            def _run(self, stocktickers: List[str], days_ago: int):
                                price_change_response = get_best_performing(stocktickers, days_ago)

                                return price_change_response

                            def _arun(self, stockticker: List[str], days_ago: int):
                                raise NotImplementedError("This tool does not support async")

                            args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput


                        def main():
                            col1, col2 = st.columns(2)
                            OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
                            model = ChatOpenAI(model="gpt-3.5-turbo-0613")

                            tools = [StockPriceTool(), StockPercentageChangeTool(),
                                    StockGetBestPerformingTool()]

                            # input
                            st.subheader("Ask a question about stocks")
                            st.caption("Example od using the agent with OpenAI functions")

                            with col1:
                                with st.form(key='my_form', clear_on_submit=True):
                                    input_message = st.text_input("Enter your question: ")
                                    submit_button = st.form_submit_button(label='Submit')
                                    if submit_button and input_message:
                                        open_ai_agent = initialize_agent(tools,
                                                                        model,
                                                                        agent=AgentType.OPENAI_FUNCTIONS,
                                                                        verbose=False)
                                        agent_answer = open_ai_agent.run(input_message)

                                        st.write(agent_answer)

                            with col2:
                                with st.expander("Show code"):
                                    st.write(tools)
                        if __name__ == "__main__":
                            main()
                """)


if __name__ == "__main__":
    main()


