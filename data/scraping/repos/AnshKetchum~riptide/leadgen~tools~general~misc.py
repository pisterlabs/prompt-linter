from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.tools.shell.tool import ShellTool
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.tools.arxiv.tool import ArxivQueryRun
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool

import platform
#from langchain.chains import create_extraction_chain - soon, ingest data from webpages

general_tools = [
    ShellTool(),
    *FileManagementToolkit().get_tools(),
    DuckDuckGoSearchRun(),
    YahooFinanceNewsTool(), 
]
