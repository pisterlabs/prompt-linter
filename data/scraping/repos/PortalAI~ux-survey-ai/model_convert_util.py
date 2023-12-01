from langchain.schema import messages
from model import chat
import json


def langchain_message_to_chat_message(message: messages.BaseMessage) -> chat.Message:
    return chat.Message(role=message.type, content=message.content)


def str_to_chat_history(chat_history_str: str) -> chat.ChatHistory:
    chat_messages: list[dict] = json.loads(chat_history_str)
    messages: list[chat.Message] = []
    for chat_message in chat_messages:
        messages.append(chat.Message(
            role=chat_message["type"],
            content=chat_message["data"]["content"],
        ))
    return chat.ChatHistory(messages=messages)
