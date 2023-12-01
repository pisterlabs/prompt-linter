import json
import logging
from datetime import datetime
from typing import Dict, List

from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import (
    BaseMessage,
    _message_from_dict,
    _message_to_dict,
    messages_from_dict,
)
from sqlalchemy.orm import Session

from expert_gpts.database import get_db_session
from expert_gpts.database.chat_message import ChatMessage as ExpertGPTsChatMessage

logger = logging.getLogger(__name__)


class MysqlFuzzySearchConfig:
    def __init__(self, distance: int = 5, limit: int = 5):
        self.distance = distance
        self.limit = limit

    def __repr__(self):
        return f"<MysqlFuzzySearchConfig distance={self.distance} limit={self.limit}>"

    def __str__(self):
        return self.__repr__()


class MysqlChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Postgres database."""

    def __init__(self, session_id: str, ai_key: str, session: Session):
        self.ai_key = ai_key
        self.session = session
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all messages from db"""
        with get_db_session() as session:
            result = session.query(ExpertGPTsChatMessage).where(
                ExpertGPTsChatMessage.session_id == self.session_id,
                ExpertGPTsChatMessage.ai_key == self.ai_key,
            )
            items = [json.loads(record.message) for record in result]
            messages = messages_from_dict(items)
            return messages

    @property
    def raw_messages(self):  # type: ignore
        """Retrieve all messages from db"""
        with get_db_session() as session:
            result = session.query(ExpertGPTsChatMessage).where(
                ExpertGPTsChatMessage.session_id == self.session_id
            )
            items = [
                {
                    "message": _message_from_dict(json.loads(record.message)),
                    "created_at": record.created_at,
                }
                for record in result
            ]
            return items

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in db"""
        with get_db_session() as session:
            jsonstr = json.dumps(_message_to_dict(message))
            session.add(
                ExpertGPTsChatMessage(
                    session_id=self.session_id,
                    message=jsonstr,
                    ai_key=self.ai_key,
                    quality=0,
                    created_at=datetime.now().isoformat(),
                )
            )
            session.commit()

    def clear(self) -> None:
        """Clear session memory from db"""

        with get_db_session() as session:
            session.query(ExpertGPTsChatMessage).filter(
                ExpertGPTsChatMessage.session_id == self.session_id
            ).delete()

    def fuzzy_search(self, search: str, distance: int = 5, limit: int = 5):
        logger.info(f"Searching for {search} in {self.session_id}")
        with get_db_session() as session:
            messages = ExpertGPTsChatMessage.search_by_message_levenstein(
                self.ai_key, self.session_id, search, session, distance, limit
            )
            items = []
            for record in messages:
                content = json.loads(json.loads(record.message))
                items.append(
                    f"At {record.created_at}, by {content['type']}: {content['data']['content']}"
                )
            return items

    def get_chats_sessions(self) -> Dict[str, datetime]:  # type: ignore
        """Retrieve all messages from db"""
        with get_db_session() as session:
            result = (
                session.query(ExpertGPTsChatMessage)
                .filter(
                    ExpertGPTsChatMessage.ai_key == self.ai_key,
                )
                .order_by(ExpertGPTsChatMessage.created_at.desc())
                .distinct()
            )
            sessions_dict = {record.session_id: record.created_at for record in result}
            return sessions_dict

    def delete_chat_session(self, session_id) -> bool:  # type: ignore
        """Delete a record by ai_key and session_id"""
        with get_db_session() as session:
            result = (
                session.query(ExpertGPTsChatMessage)
                .filter(
                    ExpertGPTsChatMessage.ai_key == self.ai_key,
                    ExpertGPTsChatMessage.session_id == session_id,
                )
                .delete()
            )
            session.commit()
            return result
