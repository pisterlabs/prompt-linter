import logging
import structlog
import json
import os
import re

from langchain.chat_models import ChatOpenAI


from langchain import PromptTemplate, LLMChain

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class HouseBot:
    def __init__(self):
        self.logger = structlog.getLogger(__name__)
        prompt_dir = 'prompts'

        human_primpt_filename = 'housebot_human.txt'
        system_prompt_filename = 'housebot_system.txt'
        default_state_filename = 'default_state.json'

        with open(f'{prompt_dir}/{system_prompt_filename}', 'r') as f:
            system_prompt_template = f.read()
        with open(f'{prompt_dir}/{human_primpt_filename}', 'r') as f:
            human_prompt_template = f.read()
        with open(f'{prompt_dir}/{default_state_filename}', 'r') as f:
            self.default_state = f.read()

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)

        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        openai_temperature = os.getenv("OPENAI_TEMPERATURE", "0")

        self.chat = ChatOpenAI(model_name=openai_model, temperature=openai_temperature)

    def strip_emojis(self, text):
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        return RE_EMOJI.sub(r'', text)

    def generate_response(self, current_state, last_state):
        self.logger.debug("let's make a request")

        chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt, self.human_message_prompt])
        # get a chat completion from the formatted messages
        chain = LLMChain(llm=self.chat, prompt=chat_prompt)
        result = chain.run(default_state=json.dumps(self.default_state, separators=(',', ':')), current_state=current_state, last_state=last_state)

        self.logger.debug(f"let's make a request: {result}")
        # print(result.llm_output)

        #strip emoji
        result = self.strip_emojis(result)
    
        return result
