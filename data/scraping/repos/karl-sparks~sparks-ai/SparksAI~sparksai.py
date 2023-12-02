import logging
from typing import AsyncIterator
from langchain.memory.chat_message_histories import FileChatMessageHistory
from SparksAI.swarm import Swarm

logger = logging.getLogger(__name__)


class SparksAI:
    def __init__(self) -> None:
        self.swarm = Swarm()

    def notice_message(self, username: str, msg: str) -> AsyncIterator:
        convo_memory = FileChatMessageHistory(f"{username}_memory.txt").messages

        message_summary = self.swarm.get_archivist(username).run(msg)
        logger.info(message_summary)

        analyst_review = self.swarm.get_analyst_agent().invoke(
            {"content": f"Context: {message_summary}\n\nUser message: {msg}"}
        )

        logger.info(analyst_review["output"])

        return self.swarm.get_conversation_agent(username).astream(
            {
                "prior_messages": message_summary,
                "analyst_message": analyst_review["output"],
                "input_message": msg,
            }
        )
