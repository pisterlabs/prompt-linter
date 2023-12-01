#!/usr/bin/python
# -*- coding: utf-8 -*-

from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
#from getpass import getpass
import os
import openai
import traceback

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
tools = []

# OpenAI APIでモデルを指定して応答を取得する
def get_response(question):
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
#        model="gpt-4",
#        model="gpt-3.5-turbo",
#        model="gpt-3.5-turbo-0613",
#        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user", "content": question }
        ]
    )
    return response["choices"][0]["message"]["content"]

class TextQa:
    def __init__(self, doc_dir: str, doc_id: str):
        self.doc_dir = doc_dir
        self.doc_id = doc_id
        self.filepath = os.path.join(self.doc_dir, self.doc_id)
    def get_answer(self, prompt: str):
        res = get_response(prompt)
        return {
            "answer": res
        }
    def qa(self, question):
        with open(self.filepath, "r") as file:
            text_data = file.read()
            prompt = f"Input Question: {question}\nInput Text Data: {text_data}\n"
            return self.get_answer(prompt)

    @staticmethod
    def get_qa(doc_dir: str, doc_id: str):
        text_qa = TextQa(doc_dir, doc_id)
        func_ptr = text_qa.qa
        return func_ptr

if __name__ == "__main__":
    import sys
    if (len(sys.argv) == 1):
        arg = input("> ")
    else:
        arg = sys.argv[1]
    if arg == "q" or arg == "quit":
        print("See you again!")
        sys.exit(0)
    try:
        ret = get_response(arg)
    except Exception as e:
        traceback_str = traceback.format_exc()
        error_message = f"ERROR: {str(e)}"
        print(traceback_str + error_message)
        sys.exit(1)

    print(ret)
