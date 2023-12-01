"""APITable Toolkit."""
from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from apitable_toolkit.tool.tool import APITableAction
from apitable_toolkit.utilities.apitable import APITableAPIWrapper
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document


class APITableToolkit(BaseToolkit):
    """APITable Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_apitable_api_wrapper(
        cls, apitable_api_wrapper: APITableAPIWrapper
    ) -> "APITableToolkit":
        actions = apitable_api_wrapper.list()
        tools = [
            APITableAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=apitable_api_wrapper,
            )
            for action in actions
        ]
        return cls(tools=tools)

    def get_retriever(self):
        tools = self.tools[2:]
        docs = [
            Document(page_content=t.description, metadata={"index": i + 2})
            for i, t in enumerate(tools)
        ]
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        return retriever

    def get_tools(self, prompt: str) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        retriever = self.get_retriever()
        docs = retriever.get_relevant_documents(prompt)
        tools = [self.tools[d.metadata["index"]] for d in docs]
        tools.append(self.tools[0])
        tools.append(self.tools[1])
        return tools
