import os
from apikey import OPENAI_API_KEY
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Text model example

llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)

llm_result = llm.generate(
    ["Write a short title for the Viblo article about new framework Langchain"]
)

print(llm_result)

# Chat model example

chat = ChatOpenAI(temperature=0)

messages = [
    SystemMessage(
        content="You are a virtual assitant can translate any text input to Vietnamese. You can auto detect the language of input text. You can return the language of input text belong the traslated text. The format is: [source language] - [translated text]"
    ),
    HumanMessage(content="I love you"),
    AIMessage(content="English - Tôi yêu bạn."),
    HumanMessage(content="どうもありがとうございます"),
]

print(chat(messages))

# Embedding model example

embeddings = OpenAIEmbeddings()
text = "This is a test document."

query_result = embeddings.embed_query(text)
print(len(query_result), type(query_result))
