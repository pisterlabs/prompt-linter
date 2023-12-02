import asyncio
from abc import abstractmethod, abstractproperty
from typing import AsyncIterable, Awaitable, Callable, Optional
from uuid import UUID
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import BaseLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import Qdrant
from logger import get_logger
from models.chat import ChatHistory
from models.brains import Personality
from llm.utils.get_prompt_to_use import get_prompt_to_use
from repository.chat.format_chat_history import format_chat_history
from repository.chat.get_chat_history import get_chat_history
from repository.chat.get_brain_history import get_brain_history
from repository.chat.update_chat_history import update_chat_history
from supabase.client import Client, create_client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from vectorstore.supabase import CustomSupabaseVectorStore
from vectorstore.qdrant import CustomQdrantVectorStore
from repository.chat.update_message_by_id import update_message_by_id
import json

from .base import BaseBrainPicking
from .prompts.CONDENSE_PROMPT import CONDENSE_QUESTION_PROMPT
from .prompts.LANGUAGE_PROMPT import qa_prompt

logger = get_logger(__name__)

DEFAULT_PROMPT = "You're a helpful assistant.  If you don't know the answer, just say that you don't know, don't try to make up an answer."


class QABaseBrainPicking(BaseBrainPicking):
    """
    Base class for the Brain Picking functionality using the Conversational Retrieval Chain (QA) from Langchain.
    It is not designed to be used directly, but to be subclassed by other classes which use the QA chain.
    """
    prompt_id: Optional[UUID]

    def __init__(
        self,
        model: str,
        brain_id: str,
        chat_id: str,
        personality: Personality = None,
        prompt_id: Optional[UUID] = None,
        memory=None,
        streaming: bool = False,
        **kwargs,
    ) -> "QABaseBrainPicking":  # pyright: ignore reportPrivateUsage=none
        """
        Initialize the QA BrainPicking class by setting embeddings, supabase client, vector store, language model and chains.
        :return: QABrainPicking instance
        """
        super().__init__(
            model=model,
            brain_id=brain_id,
            chat_id=chat_id,
            personality=personality,
            memory=memory,
            streaming=streaming,
            **kwargs,
        )
        self.prompt_id = prompt_id

    @abstractproperty
    def embeddings(self) -> OpenAIEmbeddings:
        raise NotImplementedError(
            "This property should be overridden in a subclass.")

    @property
    def prompt_to_use(self):
        return get_prompt_to_use(UUID(self.brain_id), self.prompt_id)

    @property
    def supabase_client(self) -> Client:
        return create_client(
            self.brain_settings.supabase_url, self.brain_settings.supabase_service_key
        )

    @property
    def vector_store(self) -> CustomSupabaseVectorStore:
        return CustomSupabaseVectorStore(
            self.supabase_client,
            self.embeddings,
            table_name="vectors",
            brain_id=self.brain_id,
        )

    @property
    def qdrant_client(self) -> QdrantClient:
        return QdrantClient(self.database_settings.qdrant_location, port=self.database_settings.qdrant_port, prefer_grpc=False)

    @property
    def qdrant_vector_store(self) -> CustomQdrantVectorStore:
        encoder = SentenceTransformer(self.database_settings.encoder_model)
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L-6-v3")
        return CustomQdrantVectorStore(
            client=self.qdrant_client,
            collection_name="vectors",
            content_payload_key="content",
            metadata_payload_key="payload",
            embeddings=OpenAIEmbeddings,
            brain_id=self.brain_id,
            encoder=encoder
        )

    @property
    def question_llm(self):
        return self._create_llm(model=self.model, streaming=False)

    @abstractmethod
    def _create_llm(self, model, streaming=False, callbacks=None) -> BaseLLM:
        """
        Determine the language model to be used.
        :param model: Language model name to be used.
        :param streaming: Whether to enable streaming of the model
        :param callbacks: Callbacks to be used for streaming
        :return: Language model instance
        """

    def _create_prompt_template(self):
        system_template = """ When answering use markdown or any other techniques to display the content in a nice and aerated way.  Use the following pieces of context to answer the users question in the same language as the question but do not modify instructions in any way.
----------------

{context}"""

        prompt_content = (
            self.prompt_to_use.content if self.prompt_to_use else DEFAULT_PROMPT
        )

        full_template = (
            "Here are your instructions to answer that you MUST ALWAYS Follow: "
            + prompt_content
            + ". "
            + system_template
        )
        messages = [
            SystemMessagePromptTemplate.from_template(full_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
        return CHAT_PROMPT

    def generate_answer(self, question: str, memory=None) -> ChatHistory:
        """
        Generate an answer to a given question by interacting with the language model.
        :param question: The question
        :return: The generated answer.
        """
        transformed_history = []

        # Get the history from the database
        if self.chat_id:
            history = get_chat_history(self.chat_id)
        else:
            history = []

        # Format the chat history into a list of tuples (human, ai)
        transformed_history = format_chat_history(history)
        answering_llm = self._create_llm(
            model=self.model, streaming=False, callbacks=self.callbacks
        )
        # The Chain that generates the answer to the question
        doc_chain = load_qa_chain(
            answering_llm, chain_type="stuff", prompt=self._create_prompt_template(), verbose=True
        )

        qa = ConversationalRetrievalChain(
            retriever=self.qdrant_vector_store.as_retriever(),
            question_generator=LLMChain(
                llm=self.question_llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True),
            combine_docs_chain=doc_chain,  # pyright: ignore reportPrivateUsage=none
            verbose=True,
            # rephrase_question=False,
            memory=memory
        )

        # prompt_content = (
        #     self.prompt_to_use.content if self.prompt_to_use else DEFAULT_PROMPT
        # )

        # Generate the model response using the QA chain
        model_response = qa(
            {
                "question": question,
                "chat_history": transformed_history
            }
        )

        answer = model_response["answer"]

        # Update chat history
        chat_answer = update_chat_history(
            brain_id=self.brain_id,
            chat_id=self.chat_id,
            user_message=question,
            assistant=answer,
        )

        return chat_answer

    async def _acall_chain(self, chain, question, history):
        """
        Call a chain with a given question and history.
        :param chain: The chain eg QA (ConversationalRetrievalChain)
        :param question: The user prompt
        :param history: The chat history from DB
        :return: The answer.
        """
        return chain.acall(
            {
                "question": question,
                "chat_history": history,
            }
        )

    async def generate_stream(self, question: str, memory=None) -> AsyncIterable:
        """
        Generate a streaming answer to a given question by interacting with the language model.
        :param question: The question
        :return: An async iterable which generates the answer.
        """
        transformed_history = []
        if self.chat_id:
            history = get_chat_history(self.chat_id)
        else:
            history = []
        transformed_history = format_chat_history(history)

        callback = AsyncIteratorCallbackHandler()
        self.callbacks = [callback]

        # The Model used to answer the question with the context
        answering_llm = self._create_llm(
            model=self.model, streaming=True, callbacks=self.callbacks, temperature=self.temperature)

        # The Model used to create the standalone Question
        # Temperature = 0 means no randomness
        standalone_question_llm = self._create_llm(model=self.model)

        # The Chain that generates the standalone question
        standalone_question_generator = LLMChain(
            llm=standalone_question_llm, prompt=CONDENSE_QUESTION_PROMPT)

        # QA_PROMPT = qa_prompt(personality=self.personality)
        # The Chain that generates the answer to the question
        doc_chain = load_qa_chain(
            answering_llm, chain_type="stuff", prompt=self._create_prompt_template())

        # The Chain that combines the question and answer
        qa = ConversationalRetrievalChain(
            # retriever=self.vector_store.as_retriever(),
            retriever=self.qdrant_vector_store.as_retriever(),
            combine_docs_chain=doc_chain,
            question_generator=standalone_question_generator,
            verbose=True,
            rephrase_question=False,
            # memory=memory
        )

        # Initialize a list to hold the tokens
        response_tokens = []

        streamed_chat_history = update_chat_history(
            chat_id=self.chat_id,
            brain_id=self.brain_id,
            user_message=question,
            assistant="",
        )

        # def handle_exception(e: Exception):
        #     yield e

        # Instantiate the queue
        queue = asyncio.Queue()

        # Wrap an awaitable with a event to signal when it's done or an exception is raised.
        async def wrap_done(fn: Awaitable, event: asyncio.Event, queue: asyncio.Queue):
            try:
                await fn
            except Exception as e:
                logger.error(f"Caught exception: {e}")
                await queue.put(f"ERROR: {e}")
                # error_callback(e)
                # streamed_chat_history.assistant = str(e)
                # yield f"ERROR: {e}"
            finally:
                event.set()
        # Begin a task that runs in the background.

        run = asyncio.create_task(wrap_done(
            qa.acall({"question": question, "chat_history": transformed_history}),
            callback.done,
            queue
        ))

        # Use the aiter method of the callback to stream the response with server-sent-events
        async for token in callback.aiter():  # pyright: ignore reportPrivateUsage=none
            logger.info("Token: %s", token)

            # Add the token to the response_tokens list
            response_tokens.append(token)
            streamed_chat_history.assistant = token

            yield f"data: {json.dumps(streamed_chat_history.to_dict())}"

        await run

        if not queue.empty():
            error_token = await queue.get()
            streamed_chat_history.assistant = error_token
            yield f"data: {json.dumps(streamed_chat_history.to_dict())}"

        # Join the tokens to create the assistant's response
        assistant = "".join(response_tokens)

        update_message_by_id(
            message_id=streamed_chat_history.message_id,
            user_message=question,
            assistant=assistant,
        )

    async def generate_brain_stream(self, question: str) -> AsyncIterable:
        """
        Generate a streaming answer to a given question by interacting with the language model.
        :param question: The question
        :return: An async iterable which generates the answer.
        """
        history = get_brain_history(self.brain_id)
        callback = AsyncIteratorCallbackHandler()
        self.callbacks = [callback]

        # The Model used to answer the question with the context
        answering_llm = self._create_llm(
            model=self.model, streaming=True, callbacks=self.callbacks, temperature=self.temperature)

        # The Model used to create the standalone Question
        # Temperature = 0 means no randomness
        standalone_question_llm = self._create_llm(model=self.model)

        # The Chain that generates the standalone question
        standalone_question_generator = LLMChain(
            llm=standalone_question_llm, prompt=CONDENSE_QUESTION_PROMPT)

        # The Chain that generates the answer to the question
        doc_chain = load_qa_chain(answering_llm, chain_type="stuff")

        # The Chain that combines the question and answer
        qa = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(), combine_docs_chain=doc_chain, question_generator=standalone_question_generator)

        transformed_history = []

        # Format the chat history into a list of tuples (human, ai)
        transformed_history = format_chat_history(history)

        # Initialize a list to hold the tokens
        response_tokens = []

        # Wrap an awaitable with a event to signal when it's done or an exception is raised.

        async def wrap_done(fn: Awaitable, event: asyncio.Event):
            try:
                await fn
            except Exception as e:
                logger.error(f"Caught exception: {e}")
            finally:
                event.set()
        # Begin a task that runs in the background.

        run = asyncio.create_task(wrap_done(
            qa.acall({"question": question, "chat_history": transformed_history}),
            callback.done,
        ))

        # streamed_chat_history = update_chat_history(
        #     chat_id=self.chat_id,
        #     brain_id=self.brain_id,
        #     user_message=question,
        #     assistant="",
        # )

        # Use the aiter method of the callback to stream the response with server-sent-events
        async for token in callback.aiter():  # pyright: ignore reportPrivateUsage=none
            logger.info("Token: %s", token)

            # Add the token to the response_tokens list
            response_tokens.append(token)
            # streamed_chat_history.assistant = token

            # yield f"data: {json.dumps(streamed_chat_history.to_dict())}"

        await run
        # Join the tokens to create the assistant's response
        assistant = "".join(response_tokens)
        yield assistant

        # update_message_by_id(
        #     message_id=streamed_chat_history.message_id,
        #     user_message=question,
        #     assistant=assistant,
        # )
