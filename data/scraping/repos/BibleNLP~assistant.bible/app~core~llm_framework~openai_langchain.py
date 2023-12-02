"""Implemetations for lmm_framework interface using langchain"""
import os
from typing import List, Tuple
from langchain.chat_models import ChatOpenAI

# from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# from langchain.memory import ConversationBufferMemory

from core.llm_framework import LLMFrameworkInterface
from core.vectordb import VectordbInterface
from core.vectordb.chroma4langchain import Chroma

from custom_exceptions import AccessException, OpenAIException, ChatErrorResponse
from log_configs import log


# pylint: disable=too-few-public-methods,fixme, super-init-not-called
# pylint: disable=too-few-public-methods, unused-argument, too-many-arguments, R0801


class LangchainOpenAI(LLMFrameworkInterface):
    """Uses OpenAI APIs to create vectors for text"""

    api_key: str = None
    model_name: str = None
    api_object = None
    llm = None
    chain = None
    vectordb = None

    def __init__(
        self,  # pylint: disable=super-init-not-called
        # FIXME : Ideal to be able to mock the __init__ from tests
        key: str = os.getenv("OPENAI_API_KEY", "dummy-for-test"),
        model_name: str = "gpt-3.5-turbo",
        vectordb: VectordbInterface = Chroma(),
        max_tokens_limit: int = int(
            os.getenv("OPENAI_MAX_TOKEN_LIMIT", "3052")),
    ) -> None:
        """Sets the API key and initializes library objects if any"""
        if key is None:
            raise AccessException(
                "OPENAI_API_KEY needs to be provided."
                + "Visit https://platform.openai.com/account/api-keys"
            )
        self.api_key = key
        self.model_name = model_name
        self.vectordb = vectordb
        self.api_object = ChatOpenAI
        self.api_object.api_key = self.api_key
        self.llm = self.api_object(
            temperature=0, model_name=self.model_name, openai_api_key=self.api_key
        )
        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectordb,
            # memory = memory,
            max_tokens_limit=max_tokens_limit,
            return_source_documents=True,
        )

    def generate_text(
        self, query: str, chat_history: List[Tuple[str, str]], **kwargs
    ) -> dict:
        """Prompt completion for QA or Chat reponse, based on specific documents, if provided"""
        if len(kwargs) > 0:
            log.warning(
                "Unused arguments in LangchainOpenAI.generate_text(): ", **kwargs
            )
        try:
            return self.chain({"question": query, "chat_history": chat_history})
        except ChatErrorResponse as exe:
            raise exe
        except Exception as exe:
            raise OpenAIException(
                "While generating answer: " + str(exe)) from exe
