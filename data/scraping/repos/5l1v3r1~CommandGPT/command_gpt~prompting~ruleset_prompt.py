# Full prompt with base prompt, time, memory, and historical messages

from typing import Any, Callable, List

from pydantic import BaseModel

from langchain.prompts.chat import (
    BaseChatPromptTemplate,
)
from langchain.schema import BaseMessage, SystemMessage
from langchain.vectorstores.base import VectorStoreRetriever


class RulesetPrompt(BaseChatPromptTemplate, BaseModel):
    """
    Prompt template for Ruleset Generator with base prompt, memory, and historical messages. Gets full prompt from prompt_generator.py.
    """
    ruleset_that_will: str = "conduct research and generate reports on"
    topic: str = "the effects of climate change on the economy"
    generated_ruleset: str = None
    user_feedback: str = ""
    token_counter: Callable[[str], int]
    send_token_limit: int = 3000

    def construct_full_prompt(self) -> str:
        ruleset_request = f"CommandGPT is an LLM driven application that operates in a continuous manner to achieve goals and fulfill a given purpose. The AI is prompted to use commands in order to interface with it's virtual environment.\n\nA ruleset is provided to the AI throughout it's lifecycle with the goal of keeping it on track autonomously long-term. The AI frequently loses context on information, forgets to write content, and gets stuck in repetitive cycles, so the purpose stated in the ruleset (an AI designed to...) needs to be detailed and guiding, including specific instructions around writing files and writing higher order content (the format will depend on the request).\n\nIn general, the ruleset should: \n- Instruct the AI to meticulously write content to detailed markdown (.md) files\n- be tailored to the content requested\n- be structured in a way that is easy to understand and act on for an LLM. \n- be phrased as \"You are xxx-gpt (filling in a descriptive & relevant name), an AI designed to...\" \n- should be followed by a sensible outline/breakdown of how it will do this within the context of the request & topic.\n\nGenerate an input summary that will instruct CommandGPT to {self.ruleset_that_will}:\n\n{self.topic}\n\nEnsure that the output includes only the ruleset, starting with the previously mentioned phrase \"You are xxx-gpt...\", as it will be used to kick off the AI's lifecycle."

        # If a ruleset hasn't been generated yet, just request one
        if self.generated_ruleset is None:
            return ruleset_request

        # If a ruleset has been generated, request a new one that incorporates user feedback
        new_ruleset_request = f"User: Thank you for this! Please provide an additional ruleset with the following in mind:\n- {self.user_feedback}\n...starting your response with \"You are xxx-gpt\" and providing only the ruleset with no additional text or niceties:\n"

        return new_ruleset_request

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt())
        used_tokens = self.token_counter(base_prompt.content)

        # Get relevant memory & format into message
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(
            str(previous_messages[-10:]))
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
            f"This reminds you of events from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)

        # Append historical messages if there is space
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens

        messages: List[BaseMessage] = [base_prompt, memory_message]
        messages += historical_messages
        return messages
