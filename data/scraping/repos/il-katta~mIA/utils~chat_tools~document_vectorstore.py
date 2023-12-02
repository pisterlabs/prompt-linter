from typing import Optional, List

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.llms.openai import OpenAI
from langchain.pydantic_v1 import Field
from langchain.pydantic_v1 import PrivateAttr
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.tools.base import BaseTool


# https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore

class DocumentVectorStore(BaseTool):
    """Tool that use a VectorStore"""
    _retriever: BaseRetriever = PrivateAttr()
    vectorstore: VectorStore = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._retriever = self.vectorstore.as_retriever()

    @property
    def args(self) -> dict:
        return {
            "query": {
                "type": "string",
                "description": """Question or query about/over the document, For example: 'what is the title of the document "greetings.txt" ?' or 'can you make a summary of the document "final documentation.pdf"?' """
            },
            "filename": {
                "type": "string",
                "description": """The filename of the document to be queried. For example: 'greetings.txt' or 'final documentation.pdf' """
            }
        }

    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)

    def _run(
            self,
            query: Optional[str] = None,
            filename: Optional[str] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs,
    ) -> List[Document]:
        '''

        :param str query: Question or query about/over the document, For example: 'what is the title of the document "greetings.txt" ?' or 'can you make a summary of the document "final documentation.pdf"?'
        :param str filename: The document to be queried. For example: 'greetings.txt' or 'final documentation.pdf'
        :param run_manager:
        :param kwargs:
        :return:
        '''
        if query is None:
            query = ""
        metadata = {}
        if filename:
            metadata["filename"] = filename
        return self._retriever.get_relevant_documents(
            query=query,
            callbacks=run_manager.get_child() if run_manager else None,
            metadata=metadata,
            **kwargs,
        )
