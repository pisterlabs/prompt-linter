# !/usr/bin/env python3
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import prompts.goal_message as prompts

class  GoalAgent:
    def __init__(self, type="zero_shot") -> None:
        llm = ChatOpenAI(
            temperature = 0.9
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(prompts.system())

        if type == "zero_shot":
            human_message_prompt = HumanMessagePromptTemplate.from_template(prompts.human_zero_shot())
        elif type == "few_shot":
            human_message_prompt = HumanMessagePromptTemplate.from_template(prompts.human_few_shot())

        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

        self.chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)

    def run(self, input):
        return self.chain.run(input)