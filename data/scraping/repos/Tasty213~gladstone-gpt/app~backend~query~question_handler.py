from typing import Coroutine
from fastapi import WebSocket
from opentelemetry import trace
from messageData import MessageData
from settings.chat_bot_settings import ChatbotSettings
from captcha import QuestionTooLongError, captcha_check, throw_on_long_question
from schema.message import Message
from schema.api_question import ApiQuestion
from query.llm_chain_factory import LLMChainFactory
from langchain.vectorstores.base import VectorStore


class QuestionHandler:
    def __init__(
        self,
        vector_store: VectorStore,
        settings: ChatbotSettings,
        messageDataTable: MessageData,
    ):
        self.llm_chain_factory = LLMChainFactory(
            messageDataTable, vector_store, settings
        )

    async def handle_question(self, websocket: WebSocket) -> Coroutine:
        try:
            question = await websocket.receive_json()
            captcha_check(question.get("captcha"), websocket.client)
            throw_on_long_question(question)

            chat_history: list[Message]
            chat_history = ApiQuestion.from_list(
                question.get("messages")
            ).message_history

            qa_chain = self.llm_chain_factory.make_chain(
                websocket,
                chat_history[-1],
            )

            return await qa_chain.acall(
                {
                    "question": chat_history[-1].message.content,
                    "chat_history": [
                        chat_entry.message for chat_entry in chat_history[:-1]
                    ],
                }
            )

        except QuestionTooLongError as e:
            current_span = trace.get_current_span()
            current_span.record_exception(e)
            resp = {
                "sender": "bot",
                "message": "Your question was too long to be processed, please phrase your question in less that 250 characters.",
                "type": "error",
            }
            return websocket.send_json(resp)
