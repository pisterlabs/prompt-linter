from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

_open_ai = OpenAI()
_chat_open_ai = ChatOpenAI()


class OpenAIService:
    llm: OpenAI = _open_ai
    chat_model: ChatOpenAI = _chat_open_ai
