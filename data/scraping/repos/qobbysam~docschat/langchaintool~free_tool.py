
from __future__ import annotations
import os 
import numpy as np
from abc import ABC

from logging import getLogger
from langchain.chains.base import Chain
from sklearn.feature_extraction.text import TfidfVectorizer
from paperchat.models import ChatTransaction
from customuser.models import UserProfile
from pdfpaper.models import PDFChunk, PDFFile
from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel, Field
from langchain.vectorstores.base import VectorStore
from langchain.base_language import BaseLanguageModel
from langchain.prompts import BasePromptTemplate
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain

from typing import Any, Dict, List, Optional

from .vector import DjangoVectorStore, DjangoDBDocument

logger = getLogger(__name__)

open_api_key = os.environ.get("OPENAI_API_KEY")



# class DjangoDBDocument(BaseModel):
#     page_content: str 
#     pdf_name: str 
#     page_number: str 
#     metadata: dict = Field(default_factory=dict)



class ChatDjangoDBChain(BaseConversationalRetrievalChain):

    userprofile: UserProfile
    transaction: ChatTransaction
    mode: str
    search_kwargs: dict = Field(default_factory=dict)
    top_k_docs_for_context: int = 4
    vectorstore: VectorStore = Field(alias="vectorstore")


    @property
    def _chain_type(self) -> str:
        return "chat-django-db"
    
    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[DjangoDBDocument]:
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        full_kwargs = {**self.search_kwargs, **vectordbkwargs}
        return self.vectorstore.similarity_search(
            question, k=self.top_k_docs_for_context, **full_kwargs
        )
    
    def _aget_docs(self, question: str, inputs: Dict[str, Any]) -> List[DjangoDBDocument]:
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        full_kwargs = {**self.search_kwargs, **vectordbkwargs}
        return self.vectorstore.similarity_search(
            question, k=self.top_k_docs_for_context, **full_kwargs
        )
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: DjangoVectorStore,
        userprofile: UserProfile,
        transaction: ChatTransaction,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        combine_docs_chain_kwargs: Optional[Dict] = None,
        
        **kwargs: Any,
    ) -> ConversationalRetrievalChain:
        """Load chain from LLM."""
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            **combine_docs_chain_kwargs,
        )
        condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt)

        return cls(
            vectorstore=vectorstore,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            **kwargs,
        )
