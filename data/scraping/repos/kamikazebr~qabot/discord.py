import os.path
from enum import Enum

# import inspect
import json
from langchain.schema import Document
from pydantic.fields import Field

from logger.hivemind_logger import logger
from tools.base import AgentTool
from utils.util import async_get_request
from utils.constants import VECTOR_SERVER_URL


class ConversationType(Enum):
    RAW = 0  # type=0
    SUMMARY = 1  # type=1


class DiscordTool(AgentTool):
    convo_type: ConversationType = Field(default=ConversationType.RAW, description="Conversation type")

    # override constructor
    def __init__(
            self,
            name: str,
            convo_type: ConversationType,
            description: str,
            user_permission_required: bool = False,
            **kwargs,
    ):
        super().__init__(
            name=name,
            func=self.a_conversation_search_server,
            description=description,
            user_permission_required=user_permission_required,
            **kwargs,
        )
        self.convo_type = convo_type

    async def a_conversation_search_server(self, query: str,
                                           **kwargs) -> str:
        """
            **kwargs it's used to ignore hallucination params
        """

        url = os.path.join(VECTOR_SERVER_URL, "search", str(self.convo_type.value), query)

        logger.debug(f"a_conversation_search_server->calling: {url}")
        json_response = await async_get_request(url)
        logger.debug(f"a_conversation_search_server->json_response: {json_response}")
        if json_response is None:
            return None
        list_doc = [Document(**doc) for doc in json_response]

        return self.convert_list_doc_to_str(list_doc=list_doc)

    def conversation_search(self, query: str, **kwargs) -> str:
        list_doc = self._db.similarity_search(query=query, k=5)

        return self.convert_list_doc_to_str(list_doc)

    def convert_list_doc_to_str(self, list_doc):
        new_list_doc = [
            Document(
                page_content=doc.page_content.replace("\n", " "), metadata=doc.metadata
            )
            for doc in list_doc
        ]
        # AttributeError: 'dict' object has no attribute 'page_content'
        # how build dict with page_content and metadata attributes
        # print(new_list_doc)
        l = ("\n").join(
            [
                f'message:"{doc.page_content}"\n metadata:{json.dumps(doc.metadata)}'
                for i, doc in enumerate(new_list_doc)
            ]
        )
        # do for each doc getting page content
        return l
