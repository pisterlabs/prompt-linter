import os
import random
import requests
import hashlib
from agent.tools import Tool
from typing import Union, Dict

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI


class TutorialTool(Tool):
    def __init__(self, key):
        super().__init__()
        self.invoke_label = "Tutorial"
        llm = OpenAI(model_name="gpt-3.5-turbo",
                     temperature=0.0, openai_api_key=key)
        prompt = PromptTemplate.from_template(
            "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        text = invoke_data
        result = self.chain.run(text)
        return result, 0, False, {}

    def description(self) -> str:
        return "Tutorial(text), Providing a TODO list as a toturial for the foundation model based on the given objective."
