from typing import Optional

from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.experimental.autonomous_agents.autogpt.output_parser import BaseAutoGPTOutputParser

class AutoAgent:
    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
    ) -> AutoGPT:
      llm = ChatOpenAI(temperature=0)
      search = GoogleSerperAPIWrapper()
      tools = [
          Tool(
              name = "search",
              func=search.run,
              description="useful for when you need to answer questions about current events. You should ask targeted questions"
          ),
          WriteFileTool(),
          ReadFileTool(),
      ]
      return AutoGPT.from_llm_and_tools(
          ai_name=ai_name,
          ai_role=ai_role,
          memory=memory,
          llm=llm,
          tools=tools,
          human_in_the_loop=human_in_the_loop,
          output_parser=output_parser,
          )