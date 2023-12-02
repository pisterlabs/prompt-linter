import os
from typing import Any, Callable

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import registry
from .base import BaseChat, ChatHistory, Response


TEMPLATE = '''
You are a web3 assistant. You help users use web3 apps, such as Uniswap, AAVE, MakerDao, etc. You assist users in achieving their goals with these protocols, by providing users with relevant information, and creating transactions for users.

Information to help complete your task:
{task_info}

Information about the chat so far:
{summary}

Chat History:
{history}
Assistant:'''



@registry.register_class
class SimpleChat(BaseChat):
    def __init__(self, doc_index: Any, top_k: int = 3) -> None:
        super().__init__()
        self.prompt = PromptTemplate(
            input_variables=["task_info", "summary", "history"],
            template=TEMPLATE,
        )
        self.llm = OpenAI(temperature=0.9, max_tokens=-1)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.chain.verbose = True
        self.doc_index = doc_index
        self.top_k = top_k

    def receive_input(self, history: ChatHistory, userinput: str, send: Callable) -> None:
        docs = self.doc_index.similarity_search(userinput, k=self.top_k)
        task_info = ''.join([doc.page_content for doc in docs])
        history_string = history.to_string()
        history_string += ("User: " + userinput )
        result = self.chain.run({
            "task_info": task_info,
            "summary": "",
            "history": history_string,
            "stop": "User",
        })
        result = result.strip()
        history.add_interaction(userinput, result)
        send(Response(result))
