from typing import Any, Optional

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.base import BaseOutputParser

import registry
import streaming
from .index_lookup import IndexLookupTool


TEMPLATE = '''You are a web3 assistant. You help users with answering web3-related questions. Your responses should sound natural, helpful, cheerful, and engaging, and you should use easy to understand language with explanations for jargon.

Information to help complete your task is below. Only use information below to answer the question, and create a final answer with inline citations linked to the provided source URLs. If you don't know the answer, just say that you don't know. Don't try to make up an answer. ALWAYS return a "SOURCES" part in your answer corresponding to the numbered inline citations.
---
{task_info}
---

User: {question}
Assistant:'''


@registry.register_class
class IndexAnswerTool(IndexLookupTool):
    """Tool for searching a document index and summarizing results to answer the question."""

    _chain: LLMChain

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        prompt = PromptTemplate(
            input_variables=["task_info", "question"],
            template=TEMPLATE,
        )
        new_token_handler = kwargs.get('new_token_handler')
        chain = streaming.get_streaming_chain(prompt, new_token_handler)
        super().__init__(
            *args,
            _chain=chain,
            output_description="a summarized answer with source citations",
            **kwargs
        )

    def _run(self, query: str) -> str:
        """Query index and answer question using document chunks."""
        task_info = super()._run(query)
        example = {
            "task_info": task_info,
            "question": query,
            "stop": "User",
        }
        result = self._chain.run(example)
        return result.strip()
