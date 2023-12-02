from langchain.embeddings import OpenAIEmbeddings
import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.cache import InMemoryCache
from langchain import PromptTemplate
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "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"
# openai.api_key = "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"
# api_key = "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"
# llm = OpenAI()
# chat = ChatOpenAI(openai_api_key=api_key)

# embeddings = OpenAIEmbeddings()
# text = "thsi is some normal text string that i want to embed as a vector"
# embedded_text = embeddings.embed_query(text)
# print(embedded_text )

from langchain.document_loaders import CSVLoader

loader = CSVLoader("some_data/penguins.csv")
data = loader.load()
# print([txt.page_content for txt in data])
embeddings = OpenAIEmbeddings()
embedded_docs = embeddings.embed_documents([txt.page_content for txt in data])
print(embedded_docs)
