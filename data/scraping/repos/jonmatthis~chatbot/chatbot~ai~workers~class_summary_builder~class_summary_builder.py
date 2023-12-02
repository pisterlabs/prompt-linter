import logging
import os
from datetime import datetime

import tiktoken
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from chatbot.ai.workers.class_summary_builder.class_summary_builder_prompts import \
    CLASS_SUMMARY_BUILDER_PROMPT_SYSTEM_TEMPLATE, CLASS_SUMMARY_NEW_SUMMARY_HUMAN_INPUT_PROMPT

MAX_TOKEN_COUNT = 2048
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'

load_dotenv()

logger = logging.getLogger(__name__)


class ClassSummaryBuilder:
    def __init__(self,
                 use_anthropic: bool = False,
                 ):

        self.class_summary_builder_prompt = self._create_chat_prompt()
        self._memory = ConversationBufferMemory()
        if use_anthropic:
            if os.getenv("ANTHROPIC_API_KEY") is None:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            self.llm = ChatAnthropic(temperature=0, max_tokens_to_sample=1000)
            self.llm_model = self.llm.model
            self.dollars_per_token = 0.00000163
        else:
            self.llm = ChatOpenAI(model_name='gpt-4',
                                  temperature=0,
                                  callbacks=[StreamingStdOutCallbackHandler()],
                                  streaming=True,
                                  max_tokens=4000,
                                  )
            self.llm_model = self.llm.model_name
            self.dollars_per_token = 0.00003  # gpt-4

        self._llm_chain = LLMChain(llm=self.llm,
                                   prompt=self.class_summary_builder_prompt,
                                   # memory=self._memory,
                                   verbose=True,
                                   )

    def _create_chat_prompt(self):
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            template=CLASS_SUMMARY_BUILDER_PROMPT_SYSTEM_TEMPLATE,
            input_variables={"current_summary": ""}
            )
        # system_message_prompt.prompt = system_message_prompt.prompt.format(
        #     current_summary=current_summary
        # )

        human_message_prompt = HumanMessagePromptTemplate.from_template(
            template = CLASS_SUMMARY_NEW_SUMMARY_HUMAN_INPUT_PROMPT,
            input_variables = ["new_student_summary"]
        )
        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

    async def update_class_summary_based_on_new_student_summary(self,
                                                                current_class_summary: str,
                                                                new_student_summary: str,
                                                                ) -> str:

        return await self._llm_chain.arun(
            current_summary=current_class_summary,
            new_student_summary=new_student_summary,
        )

def time_since_last_summary(class_summary_entry):
    previous_summary_datetime = datetime.strptime(class_summary_entry["class_summary"]["created_at"],
                                                  DATE_FORMAT)
    current_time = datetime.now()
    time_since_last_summary = current_time - previous_summary_datetime
    time_since_last_summary_in_hours = time_since_last_summary.total_seconds() / 3600
    return time_since_last_summary_in_hours


def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

