"""Search in Pinecone using namespaces and metadata filters, and support multiple queries by namespace as part of a single retrieval call."""
from __future__ import annotations
import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
# from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain

import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import Extra, Field, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains.base import Chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMessage, BaseRetriever, Document
from langchain.vectorstores.base import VectorStore
import json

from loguru import logger
from collections import defaultdict
import regex as re

from spiritualchat.api_functions.combine_docs_chain import NamespaceStuffDocumentsChain
from spiritualdata_utils import mongo_query_db, mongo_connect_db


mongo = mongo_connect_db(database_name='spiritualdata')

# Depending on the memory type and configuration, the chat history format may differ.
# This needs to be consolidated.
CHAT_TURN_TYPE = Union[Tuple[str, str], BaseMessage]

class OptionalContentDocument(Document):
    page_content: Optional[str]

class PineconeNamespaceSearchRetriever(BaseRetriever, BaseModel):
    embeddings: Embeddings
    index: Any
    top_k: int = 4
    alpha: float = 0.5

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get_relevant_documents(
        self,
        namespace_queries: dict,
        metadata_filter: dict = None,
        callbacks: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[Document]]:
        """
        Args:
            namespace_queries (dict): Key is namespace in the Pinecone index. Value is a list of strings (queries to embed).
            metadata_filter (dict): Key is the metadata field associated with each Pinecone vector, and value is the filter applied on that field (see https://docs.pinecone.io/docs/metadata-filtering).
            callbacks (CallbackManagerForRetrieverRun): A manager for callbacks to be used during execution.

        Returns:
            namespace_docs (dict): Key is namespace in the Pinecone index. Value is a list of Document objects.
        """
        global mongo
        namespace_docs = defaultdict(list)

        for namespace, queries in namespace_queries.items():
            namespace = namespace.replace('_queries', '')
            pinecone_ids = []
            for query in queries:
                dense_vec = self.embeddings.embed_query(query)
                result = self.index.query(
                    vector=dense_vec,
                    top_k=self.top_k,
                    include_metadata=True,
                    namespace=namespace,
                    filter=metadata_filter
                )
                for res in result["matches"]:
                    pinecone_ids.append(res["id"])
                    res["metadata"]["id"] = res["id"]
                    namespace_docs[namespace].append(
                        OptionalContentDocument(page_content=None, metadata=res["metadata"])
                    )

            # Query MongoDB for the 'name' and 'text' fields of the collected Pinecone IDs
            mongo_results = mongo_query_db(
                query_type="find",
                mongo_object=mongo,
                query={"pinecone_id": {"$in": pinecone_ids}},
                collection=namespace,
                projection={"name": 1, "text": 1, "pinecone_id": 1},
            )

            # Create a dictionary to map Pinecone IDs to their corresponding MongoDB results
            mongo_dict = {result["pinecone_id"]: result for result in mongo_results}

            # Update the placeholder Documents with the 'name' and 'text' fields from the MongoDB results
            for document in namespace_docs[namespace]:
                pinecone_id = document.metadata["id"]
                mongo_result = mongo_dict[pinecone_id]
                document.page_content = f'{mongo_result["name"]}\n{mongo_result["text"]}'

        if callbacks:
            callbacks.on_result(namespace_docs)

        return namespace_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

def _get_chat_history(chat_history: List[CHAT_TURN_TYPE]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        try:
            if isinstance(dialogue_turn, str):
                buffer += "\n" + dialogue_turn
            elif isinstance(dialogue_turn, BaseMessage):
                role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
                buffer += f"\n{role_prefix}{dialogue_turn.content}"
            elif isinstance(dialogue_turn, tuple):
                human = "Human: " + dialogue_turn[0]
                ai = "Assistant: " + dialogue_turn[1]
                buffer += "\n" + "\n".join([human, ai])
            else:
                raise ValueError(
                    f"Unsupported chat history format: {type(dialogue_turn)}."
                    f" Full chat history: {chat_history} "
                )
        except TypeError:
            raise TypeError(
                    f"Unsupported chat history format: {type(dialogue_turn)}."
                    f" Full chat history: {chat_history} "
                )
    return buffer

class NamespaceSearchConversationalRetrievalChain(ConversationalRetrievalChain):
    """Chain for chatting with an index."""

    combine_docs_chain: NamespaceStuffDocumentsChain
    question_generator: LLMChain
    output_key: str = "answer"
    return_source_documents: bool = True
    return_generated_question: bool = True
    get_chat_history: Optional[Callable[[CHAT_TURN_TYPE], str]] = None
    """Return the source documents."""

    def transform_query(self, ai_response: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Transforms AI response containing namespace queries into a Python dict.

        Args:
            ai_response (str): AI response potentially containing namespace queries.

        Returns:
            Dict[str, List[str]]: Python dictionary mapping namespaces to a list of queries.
        """
        # Extract JSON part from AI response
        title = None
        try:
            # First, try to parse the whole response as JSON
            namespace_queries = json.loads(ai_response)
            title = namespace_queries.pop('title')
        except json.JSONDecodeError:
            # If that fails, try to extract JSON part using a non-greedy regex
            try:
                json_part = re.search('{.*?}', ai_response).group(0)
                namespace_queries = json.loads(json_part)
                title = namespace_queries.pop('title')
            except (AttributeError, json.JSONDecodeError):
                print(f"Error on AI response: {error}\nCouldn't extract JSON from: {ai_response}")
                return {}
        return title, namespace_queries

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        callbacks = _run_manager.get_child()
        new_question = self.question_generator.run(
            question=question, chat_history=chat_history_str, callbacks=callbacks
        )

        title, namespace_queries = self.transform_query(new_question)
        namespace_docs = self._get_docs(namespace_queries, inputs, run_manager=_run_manager)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer = self.combine_docs_chain.run(
            input_documents=namespace_docs, callbacks=_run_manager.get_child(), **new_inputs
        )

        output: Dict[str, Any] = {self.output_key: answer}
        if self.return_source_documents:
            output["source_documents"] = namespace_docs
        if self.return_generated_question:
            output["generated_question"] = new_question
        if title:
            output["title"] = title
        return output

    def _reduce_tokens_below_limit(self, namespace_docs: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
        """
        This method reduces the total number of tokens in all retrieved documents across all namespaces to be 
        below the specified maximum tokens limit. This is done by iteratively removing the document with the 
        highest token count from each namespace's list of documents until the total token count is below the limit 
        or no more documents can be removed.
        
        Args:
            namespace_docs (dict): A dictionary where keys are namespaces and values are lists of Document objects 
                                   retrieved for each namespace.

        Returns:
            namespace_docs (dict): A dictionary similar to the input but with the lists of Document objects modified 
                                   to ensure the total token count across all documents is below the maximum limit.
        """
        if not self.max_tokens_limit:
            return namespace_docs

        # Get tokens per namespace and sort by namespace
        tokens = {
            namespace: [
                self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content)
                for doc in docs
            ]
            for namespace, docs in namespace_docs.items()
        }

        # Calculate total token count
        token_count = sum(sum(tkns) for tkns in tokens.values())
        for namespace, docs in namespace_docs.items():
            if not all(isinstance(doc, Document) for doc in docs):
                logger.error(f"Found non-Document item in docs for namespace {namespace}")
        namespaces = list(namespace_docs.keys())
        while token_count > self.max_tokens_limit:
            for namespace in namespaces:
                if token_count <= self.max_tokens_limit:
                    break
                if tokens[namespace]:
                    token_count -= tokens[namespace].pop()
                    namespace_docs[namespace].pop()
        for namespace, docs in namespace_docs.items():
            if not all(isinstance(doc, Document) for doc in docs):
                logger.error(f"Found non-Document item in docs for namespace {namespace}")
        return namespace_docs

    def _get_docs(
        self,
        namespace_queries: Dict[str, List[str]],
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> Dict[str, List[Document]]:
        """
        This method retrieves the relevant documents for each namespace based on the queries and then applies 
        token reduction to ensure the total token count across all retrieved documents is below the specified limit.
        
        Args:
            namespace_queries (dict): A dictionary where keys are namespaces and values are lists of queries to be run for each namespace.
            inputs (dict): A dictionary of inputs required for the retriever.
            run_manager (CallbackManagerForChainRun): A manager for callbacks to be used during execution.

        Returns:
            namespace_docs (dict): A dictionary where keys are namespaces and values are lists of Document objects retrieved 
                                   for each namespace, adjusted to ensure the total token count across all documents is below 
                                   the maximum limit.
        """
        namespace_docs = self.retriever.get_relevant_documents(
            namespace_queries, callbacks=run_manager.get_child()
        )

        # Apply token reduction across all namespaces
        namespace_docs = self._reduce_tokens_below_limit(namespace_docs)
        return namespace_docs

NamespaceSearchConversationalRetrievalChain.update_forward_refs()