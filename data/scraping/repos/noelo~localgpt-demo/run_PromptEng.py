import os
import openai
import chainlit as cl
from chainlit import AskUserMessage, Message, on_chat_start
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY



SYSTEM_TEMPLATE = """Use the following pieces of context to answer the users question. Keep the answer short and concise and don't explain the generated text.
If you don't know the answer, just say that you don't know, don't try to make up an answer.


Example of your response should be:

```
The answer is foo
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@on_chat_start
def main():
        Message(
            content=f"Generate kubernetes resources",
        ).send()


@cl.langchain_factory
def load_model():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    local_llm = OpenAI(temperature=0.0)
    return local_llm


@cl.langchain_postprocess
def process_response(res):
    answer = res["result"]
    cl.Message(content=answer).send()