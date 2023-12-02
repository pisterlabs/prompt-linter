from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI


def format_answer(question, answer):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    message = f'Question:{question} Answer: {answer}'

    messages = [
        SystemMessage(content="Simply make the given answer into a single sentence"),
        HumanMessage(content=message)
    ]

    response = chat(messages)
    return response.content
