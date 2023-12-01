import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)




os.environ["OPENAI_API_KEY"] = "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"

chat = ChatOpenAI(temperature=0)


chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

AIMessage(content="J'aime programmer.", additional_kwargs={})
