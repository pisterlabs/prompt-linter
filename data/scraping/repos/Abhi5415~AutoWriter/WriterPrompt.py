from __future__ import annotations

from langchain.schema import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever

import time
from typing import Any, Callable, List, Union

from pydantic import BaseModel

from langchain.prompts.chat import (
    BaseChatPromptTemplate,
)
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever
from utils.StageReturnType import StageReturnType
from BaseContent import BaseContent
from prompts.PromptGenerator import get_prompt


class WriterPrompt(BaseChatPromptTemplate, BaseModel):
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    def construct_full_prompt(self, content: BaseContent, feedback: Union[str, None]) -> str:
        prompt_start = (
            "If you have completed all your tasks, make sure to "
            'use the "finish" command.'
        )
        # Construct full prompt
        full_prompt = (
            f"You are a technology expert who is writing an outline for a blog post for the internet. \n{prompt_start}\n"
        )
        full_prompt += f"You are responsible for turning the outline and research into a well written and clear article that will be presented as a blog post.\n"

        # goals
        full_prompt += f"The article's topic is: {content.title}\n"
        full_prompt += f"The goals of the article are as follows: {content.goals}\n"
        full_prompt += f"The audience of the article are {content.audience}\n"
        full_prompt += f"The article's tone should be {content.tone}\n"
        full_prompt += f"The article's length should be about 600 words\n"
        full_prompt += f"Focus on real-world examples and use cases. This is a blog post for the internet, not a technical paper.\n"
        full_prompt += f"Include captions for images you want an unskilled human artist to draw in parenthesis like so: [draw a picture of a server interacting with a client machine] \n"
        full_prompt += "\n\n"

        # outline
        full_prompt += "You are given the following outline:\n"
        full_prompt += f"{content.outline}\n"
        full_prompt += "\n\n"

        # research results
        full_prompt += "You are given the following research:\n"
        for question, answer in content.research.items():
            full_prompt += f"Question: {question}: {answer}\n"
        full_prompt += f"If you feel this is insufficient information to write this article then request more information using the appropriate tool.\n"
        full_prompt += "\n\n"

        # feedback
        if feedback:
            full_prompt += f"You've received the following feedback on your work so far: {feedback}\n\n"

#         # directives
#         full_prompt += """
# You should ask the human for feedback if you're unclear on what the article should be about.
# You're finished when you've written an article and have received positive feedback.
# You should be critical and skeptical of the research and outline given to you. If you feel it is insufficient or contains an error, provide feedback with the "Outline" tool.
#         """

        full_prompt += f"\n\n{get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs["content"], kwargs["feedback"]))
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}"
        )
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(
            time_prompt.content
        )
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory]
        )
        while used_tokens + relevant_memory_tokens > 2500 and len(relevant_memory) > 0:
            print("here")
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory]
            )
        content_format = (
            f"This reminds you of these events "
            f"from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages
