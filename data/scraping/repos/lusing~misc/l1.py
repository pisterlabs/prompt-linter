from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)

result = chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

print(result)