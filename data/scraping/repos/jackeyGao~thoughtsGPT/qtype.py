from typing import List
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from thoughts_gpt.core.prompts import STUFF_PROMPT
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from thoughts_gpt.core.embedding import FolderIndex
from thoughts_gpt.core.prompts import QTYPE_PROMPT
from thoughts_gpt.core.const import SUGGESTED_QUESTION_PREFIX
from thoughts_gpt.core.const import DOCUMENT_SUMMARIES_SOURCE

_VALID_QTYPE = ["qa", "summarization"]

class AnswerWithSources(BaseModel):
    qtype: str


def query_qtype(
    query: str,
    llm: BaseChatModel,
) -> AnswerWithSources:
    """Queries a folder index for an answer.

    Args:
        query (str): The query to search for.
        llm (str): The model to use for the answer generation.
        **model_kwargs (Any): Keyword arguments for the model.

    Returns:
        AnswerWithSources: The answer and the source documents.
    """

    chain = LLMChain(llm=llm, prompt=QTYPE_PROMPT)

    result = chain(
        { "question": query}, 
        return_only_outputs=True
    )

    if result["text"] not in _VALID_QTYPE:
        result["text"] = _VALID_QTYPE[0]

    return AnswerWithSources(qtype=result.get('text', _VALID_QTYPE[0]))

