import logging
from typing import List, Optional, SupportsIndex, cast
from langchain.chains.query_constructor.ir import StructuredQuery

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import (
    BaseMessage,
    _message_to_dict,
    messages_from_dict,
    messages_to_dict,
)

from langchain.memory import DynamoDBChatMessageHistory
from langchain.chains import sql_database
from sympy.parsing.sympy_parser import _T
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DynamoDBChatMessageHistoryNew(DynamoDBChatMessageHistory):
    messages: List[BaseMessage] = []

    def __init__(self, table_name: str, session_id: str, endpoint_url: Optional[str] = None):
        super().__init__(table_name, session_id, endpoint_url)
        self.messages = MessageStore.from_chat_history(self)

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in DynamoDB"""
        self.messages.append(message)

    def clear(self) -> None:
        super().clear()
        self.messages = []


class MessageStore(list):
    def __init__(self, chat_history: DynamoDBChatMessageHistoryNew, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_history = chat_history

    def append(self, message):
        from botocore.exceptions import ClientError

        messages = messages_to_dict(self)
        _message = _message_to_dict(message)
        messages.append(_message)

        try:
            self.chat_history.table.put_item(
                Item={"SessionId": self.chat_history.session_id, "History": messages}
            )
            super().append(message)
        except ClientError as err:
            logger.error(err)

    def pop(self, __index: SupportsIndex = ...) -> _T:
        from botocore.exceptions import ClientError

        messages = messages_to_dict(self)
        messages.pop(__index)

        try:
            self.chat_history.table.put_item(
                Item={"SessionId": self.chat_history.session_id, "History": messages}
            )
            return super().pop(__index)
        except ClientError as err:
            logger.error(err)

    @classmethod
    def from_chat_history(cls, chat_history: DynamoDBChatMessageHistoryNew):
        from botocore.exceptions import ClientError

        response = None
        try:
            response = chat_history.table.get_item(Key={"SessionId": chat_history.session_id})
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("No record found with session id: %s", chat_history.session_id)
            else:
                logger.error(error)

        if response and "Item" in response:
            items = response["Item"]["History"]
        else:
            items = []

        messages = messages_from_dict(items)
        return cls(chat_history, messages)


class SelfQueryRetrieverNew(SelfQueryRetriever):
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query.

                Args:
                    query: string to find relevant documents for

                Returns:
                    List of relevant documents
                """
        inputs = self.llm_chain.prep_inputs({"query": query})
        structured_query = cast(
            StructuredQuery,
            self.llm_chain.predict_and_parse(
                callbacks=run_manager.get_child(), **inputs
            ),
        )
        if self.verbose:
            print(structured_query)
        new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
            structured_query
        )
        print(f"struct query: {structured_query}")
        if structured_query.limit is not None:
            if structured_query.limit < 4:
                new_kwargs["k"] = 4
            else:
                new_kwargs["k"] = structured_query.limit
        else:
            new_kwargs["k"] = 10

        if self.use_original_query:
            new_query = query

        print(new_query)

        search_kwargs = {**self.search_kwargs, **new_kwargs}
        docs = self.vectorstore.search(new_query, self.search_type, **search_kwargs)
        return docs
