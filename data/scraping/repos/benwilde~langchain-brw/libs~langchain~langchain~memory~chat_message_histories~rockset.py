import os
import json
import logging
import rockset 
from typing import List

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

logger = logging.getLogger(__name__)

class RocksetChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Postgres database."""

    def __init__(
        self,
        session: str,
        collection: str = "chatbot_memory",
    ):
      
        self.session_id = session
        self.collection_name = collection

        # get the Rockset API key
        try:
            rockset_api_key = os.environ["ROCKSET_API_KEY"]
        except KeyError:
            logger.error(error)   
        
        # setup the Rockset client 
        try:
            self.rs = rockset.RocksetClient(rockset.Regions.use1a1, rockset_api_key)
        except rockset.ApiException as error:
            logger.error(error)

        # should we create catalog / table here if one does not exist?

    def _create_table_if_not_exists(self) -> None:

        # In the case of Rockset we should be checking for a collection
        # Assumes use of Common's workspace (document this)

        try:
            self.rs.Collection.retrieve(self.collection_name, workspace='commons')
        except rockset.ApiException as error:
            try:
                self.rs.Collection.create(self.collection_name, workspace="commons")
            except rockset.ApiException as error:
                logger.error(error)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        
        """Retrieve the messages from Rockset"""
        """ Requires "id" to be an autoincrement ID to maintain message sequence"""
        
        try:
            res = self.rs.sql(query="SELECT chat_row FROM :collection_name WHERE session_id = " + self.session_id + " order by _event_time", params={"collection_name", self.collection_name})
        except rockset.ApiException as e:
            print("Exception when querying: %s\n" % e)

        items = [record["chat_row"] for record in res]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        
        """Append the message to the record in Rockset"""
        """ session_id is used to uniquely identify one user session's messages""" 

        try:
            self.rs.Documents.add_documents(
            collection="memory_store",
            data=[{"chat_row":json.dumps(_message_to_dict(message))}, {"session_id":self.session_id}],
            )
        except rockset.ApiException as error:
            logger.error(error)   

    def clear(self) -> None:
        """Clear session memory from Rockset"""
        try:
            res = self.rs.sql(query="delete * FROM :collection_name WHERE session_id = " + self.session_id, params={"collection_name", self.collection_name})
        except rockset.ApiException as e:
            print("Exception when querying: %s\n" % e)

    def __del__(self) -> None:
      """Clean up the connection to Rockset"""
      """ Not clear that this is requireed """
       