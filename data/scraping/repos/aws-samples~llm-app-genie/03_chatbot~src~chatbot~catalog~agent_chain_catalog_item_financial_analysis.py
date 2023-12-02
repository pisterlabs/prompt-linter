""" Module that contains an abstract base class that represents an agent chain item.
"""
from dataclasses import dataclass
from langchain.tools import BaseTool
from .agent_chain_catalog_item import AgentChainCatalogItem

AGENT_CHAIN_FINANCIAL_ANALYSIS_NAME = "Financial Analysis"


@dataclass
class FinancialAnalysisAgentChainItem(AgentChainCatalogItem):
    """Abstract base class that represents an agent chain for financial analysis."""

    def __init__(self):
        super().__init__(AGENT_CHAIN_FINANCIAL_ANALYSIS_NAME)

    def get_prompt_path(self) -> str:
        return 'prompts/anthropic_claude_agent_financial_analyzer.yaml'

    def get_instance(self) -> BaseTool:
        from .agent_tools_catalog_item import get_stock_price, get_recent_stock_news, get_financial_statements
        from langchain.utilities import DuckDuckGoSearchAPIWrapper
        from langchain.agents import Tool

        search_duckduckgo = DuckDuckGoSearchAPIWrapper()
        
        agent_chain = [
            Tool(
                name="get stock data",
                func=get_stock_price,
                description="Use when you are asked to evaluate or analyze a stock. This will output historic share price data for the last 60 days. You should input the stock ticker to it"
            ),
            Tool(
                name="DuckDuckGo Search",
                func=search_duckduckgo.run,
                description="Use this to fetch the stock ticker, you can also get recent stock related news. Dont use it for any other analysis or task"
            ),
            Tool(
                name="get recent news",
                func=get_recent_stock_news,
                description="Use this to fetch recent news about stocks"
            ),
            Tool(
                name="get financial statements",
                func=get_financial_statements,
                description="Use this to get financial statement of the company. With the help of this data companys historic performance can be evaluated. You should input stock ticker to it"
            )
        ]
        
        return agent_chain
