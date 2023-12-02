
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from typing import Optional, Type

#Import our Stock Adapter
from chainge.api.api import stock_api
from typing import Union, Tuple, Dict

'''

Tool Structure -->

1. Agent first gets a list of all financial attributes in object
2. Agent passess input to the tool of all the attributes it needs:

'''

class StockProbeInput(BaseModel):
    anything: str = Field()

class StockFundamentalsProbeTool(BaseTool):
    name = "stock_fundamentals_probe"

    description = """Use this tool BEFORE the stock fundamentals tool to see what attributes the fundamentals tool offers. Whatever attributes you see here
    must be used VERBATIM in the stock fundamentals tool. You won't have to pass any inputs here, but keep track of the values from the list that you'll need to answer your question. 
    """

    args_schema: Type[BaseModel] = StockProbeInput 

    #Ensure no arguments are passed in
    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}

    def _run(
        self, run_manager = None
    ) -> str:
        """Use the tool."""
        return f'Here is a list of all the fundamental attributes: {stock_api.fundamentals_lookup()}'

    async def _arun(
        self, stock_ticker_name: str, run_manager = None
    ) -> str:
        """Use the tool asynchronously."""
        return f'Here is a list of all the fundamental attributes: {stock_api.fundamentals_lookup()}'



class StockFundamentalsInput(BaseModel):
    stock_ticker_name: str = Field()
    requested_metrics: str = Field()


class StockFundamentalsTool(BaseTool):
    name = "stock_fundamentals_post_probe_data"

    description = """Use this tool after the stock probe to answer basic questions about stocks, including pricing, average trading volume, volatility (beta), cash flow, and 
    price highs and low
    
    input: stock ticker; AAPL, GOOG, and so on. REMEMBER TO USE THE STOCK PROBE TOOL BEFORE THIS!
    """
    args_schema: Type[BaseModel] = StockFundamentalsInput

    def _run(
        self, stock_ticker_name: str, requested_metrics: str, run_manager = None
    ) -> str:
        """Use the tool."""

        return stock_api.fundamentals(stock_ticker_name, requested_metrics)

    async def _arun(
        self, stock_ticker_name: str, run_manager = None
    ) -> str:
        """Use the tool asynchronously."""
        return stock_api.fundamentals(stock_ticker_name, requested_metrics)
