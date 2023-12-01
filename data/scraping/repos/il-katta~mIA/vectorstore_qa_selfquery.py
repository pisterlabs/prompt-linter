import json
from typing import Optional, Any

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.pydantic_v1 import PrivateAttr
from langchain.schema import BaseRetriever
from langchain.tools import BaseTool
from langchain.tools.vectorstore.tool import BaseVectorStoreTool
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

__all__ = ["VectorstoreQASelfQuery"]

# https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/
# https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/chroma_self_query


class VectorstoreQASelfQuery(BaseVectorStoreTool, BaseTool):
    _retriever: BaseRetriever = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(verbose=True, **kwargs)


        metadata_infos = [
            AttributeInfo(
                name="filename",
                description="Name of the file",
                type="string",
            ),
        ]

        self._retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents="files uploaded by user",
            metadata_field_info=metadata_infos,
            verbose=True,
            enable_limit=True,
        )

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name} and the sources "
            "used to construct the answer. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            " Input should be a fully formed question. "
            "Output is a json serialized dictionary with keys `answer` and `sources`. "
        )
        return template.format(name=name, description=description)

    def __call__(
            self,
            question: str,
            filename: Optional[str] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs: Any,
    ) -> str:
        '''
        :param str question: question or query about/over the document, For example: 'what is the title of the document "greetings.txt" ?' or 'can you make a summary of the document "final documentation.pdf"?'
        :param str filename: filename of the document, For example: 'greetings.txt' or 'final documentation.pdf'
        :param run_manager:
        :param kwargs:
        :return:
        '''
        return self._run(question=question, filename=filename, run_manager=run_manager, **kwargs)

    def _run(
            self,
            question: str,
            filename: Optional[str] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs: Any,
    ) -> str:
        chain = RetrievalQAWithSourcesChain.from_chain_type(self.llm, retriever=self._retriever, verbose=True)
        return json.dumps(
            chain(
                {
                    chain.question_key: json.dumps({
                        chain.question_key: question,
                        'filename': filename,
                        **kwargs
                    }),
                },
                return_only_outputs=True,
                callbacks=run_manager.get_child() if run_manager else None,
            )
        )
