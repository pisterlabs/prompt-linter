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


class OutlinerPrompt(BaseChatPromptTemplate, BaseModel):
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    def construct_full_prompt(self, content: BaseContent, feedback: Union[str, None]) -> str:
        prompt_start = (
            "Your decisions must always be made independently "
            "Play to your strengths as an LLM and pursue simple "
            "strategies with no legal complications.\n"
            "If you have completed all your tasks, make sure to "
            'use the "finish" command.'
        )
        # Construct full prompt
        full_prompt = (
            f"You are a technology expert who is writing an outline for a blog post for the internet. \n{prompt_start}\n"
        )

        # goals
        full_prompt += f"The article's title should be something like {content.title}\n"
        full_prompt += f"The goals of the article are as follows: {content.goals}\n"
        full_prompt += f"The audience of the article are {content.audience}\n"
        full_prompt += f"The article's tone should be {content.tone}\n"
        full_prompt += f"The article's length should be about 600 words\n"
        full_prompt += f"You are only writing an outline for the article based on the research. Do not attempt to write the article. Simply write the outline in a logical manner and include all the research.\n"
        full_prompt += f"Do not 'finish' until the human has given you positive feedback.\n"
        full_prompt += f"The benefits of a technology are often shown by comparing it to its alternatives. Show the alternatives and contrast the technologies discussed in the article to show their benefits.\n"
        full_prompt += f"Focus on real-world examples and use cases. This is a blog post for the internet, not a technical paper.\n"
        full_prompt += "\n\n"

        # research results
        if len(content.research) > 0:
            full_prompt += "You are given the following guiding questions for the outline:\n"
            for question, answer in content.research.items():
                full_prompt += f"Question: {question}: {answer}\n"
            full_prompt += f"If you feel this is insufficient information to write this article then request more information using the appropriate tool.\n"
            full_prompt += "\n\n"

        # feedback
        if feedback:
            full_prompt += f"You've received the following feedback on your work so far: {feedback}\n\n"

        # directives
        full_prompt += """
You should ask the human for feedback if you're unclear on what the article should be about.
You're finished when you've written a concise outline and have received positive feedback.
You should be critical and skeptical of the research given to you. If you feel it is insufficient or contains an error, provide feedback with the "Research" tool.
Your outline should be highly detailed and integrate all the information from the research in bullet points.
        """

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
        while used_tokens + relevant_memory_tokens > 2500:
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
