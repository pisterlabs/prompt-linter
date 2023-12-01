from langchain import ArxivAPIWrapper
import arxiv as arxiv_module
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool
from pydantic import BaseModel, Field
arxiv_wrapper = ArxivAPIWrapper()
class ArxivInput(BaseModel):
    preprint: str = Field()

arxiv_doc_tool = Tool.from_function(
    name="arxiv",
    func=arxiv_wrapper.run,
    args_schema=ArxivInput,
    description="Use this tool to search for research papers in arxiv pre-print repository.",
    return_direct=True
)
