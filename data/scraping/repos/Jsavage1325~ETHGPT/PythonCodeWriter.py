"""
A file which contains a tool to write Python code. The tool is just a wrapper for a LLM to write python code
"""
from typing import Optional

from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.llms import OpenAI
from langchain.tools import BaseTool


class PythonCodeWriter(BaseTool):
    name = "python_code_writer"
    description = "Python expert who will write code, using previous context from a conversation also."

    def _run(self, query: str) -> str:
        """
        Call an LLM, using an engineered prompt, a query and previous chat history as context.
        """
        print(query)
        llm = OpenAI(temperature=0.0)
        prompt = f"""
        Write Python code to answer the following question. Return only Python code.
        {query}
        """
        return llm(prompt)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("airstack_doc_search does not support async")



