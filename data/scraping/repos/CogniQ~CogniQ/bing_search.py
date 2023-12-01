from __future__ import annotations
from typing import *
import logging

logger = logging.getLogger(__name__)
import asyncio
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

from haystack.agents import Agent, Tool
from haystack.agents.base import ToolsManager
from haystack.nodes import PromptNode

from cogniq.config import OPENAI_API_KEY, OPENAI_MAX_TOKENS_RESPONSE
from cogniq.personalities import BasePersonality
from cogniq.slack import CogniqSlack
from cogniq.openai import (
    system_message,
    user_message,
    assistant_message,
    message_to_string,
    CogniqOpenAI,
)

from .prompts import agent_prompt
from .custom_web_qa_pipeline import CustomWebQAPipeline


class BingSearch(BasePersonality):
    def __init__(
        self,
        *,
        cslack: CogniqSlack,
        copenai: CogniqOpenAI,
    ):
        super().__init__(cslack=cslack, copenai=copenai)
        self.web_qa_tool = Tool(
            name="Search",
            pipeline_or_node=CustomWebQAPipeline(),
            description="useful for when you need to Google questions.",
            output_variable="answers",
        )

        self.agent_prompt_node = PromptNode(
            "gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            max_length=OPENAI_MAX_TOKENS_RESPONSE,
            stop_words=["Observation:"],
        )

    @property
    def description(self) -> str:
        return "I perform extractive generation of answers from Bing search results."

    @property
    def name(self) -> str:
        return "Bing Search"

    async def ask_directly(
        self,
        *,
        q: str,
        message_history: List[Dict[str, str]],
        context: Dict[str, Any],
        stream_callback: Callable[..., None] | None = None,
        reply_ts: str | None = None,
    ) -> str:
        ask_response = await self.ask(
            q=q,
            message_history=message_history,
            context=context,
            stream_callback=stream_callback,
        )
        transcript = ask_response["response"]["transcript"]
        transcript_summary = await self.copenai.summarizer.ceil_prompt(transcript)
        return transcript_summary

    async def ask(
        self,
        *,
        q: str,
        message_history: List[Dict[str, str]],
        context: Dict[str, Any],
        stream_callback: Callable[..., None] | None = None,
        reply_ts: str | None = None,
        thread_ts: str | None = None,
    ) -> Dict[str, Any]:
        if message_history is None:
            message_history = []

        history_augmented_prompt = await self._get_history_augmented_prompt(
            q=q,
            message_history=message_history,
            context=context,
        )

        loop = asyncio.get_event_loop()
        with PoolExecutor() as executor:
            agent_response_task = loop.run_in_executor(
                executor,
                self._agent_run,
                history_augmented_prompt,
                stream_callback,
            )
            agent_response = await agent_response_task
        final_answer = agent_response["answers"][0]
        logger.debug(f"final_answer: {final_answer}")
        final_answer_text = final_answer.answer
        if not final_answer_text:
            transcript = agent_response["transcript"]
            summarized_transcript = await self.copenai.summarizer.summarize_content(transcript, OPENAI_MAX_TOKENS_RESPONSE)
            final_answer_text = summarized_transcript
        return {"answer": final_answer_text, "response": agent_response}

    async def _get_history_augmented_prompt(self, *, q: str, message_history: List[Dict[str, str]], context: Dict[str, Any]) -> str:
        """
        Returns a prompt augmented with the message history.
        """
        # bot_id = await self.cslack.openai_history.get_bot_user_id(context=context)
        bot_name = await self.cslack.openai_history.get_bot_name(context=context)

        # if the history is too long, summarize it
        message_history = self.copenai.summarizer.ceil_history(message_history)

        # Set the system message
        message_history = [system_message(f"Hello, I am {bot_name}. I am a slack bot that can answer your questions.")] + message_history

        # if prompt is too long, summarize it
        short_q = await self.copenai.summarizer.ceil_prompt(q)

        logger.info("short_q: " + short_q)

        message_history_string = ("\n\n".join([message_to_string(message) for message in message_history]),)
        prompt = f"""Conversation history: {message_history_string}

        Query: {short_q}"""
        history_augmented_prompt = await self.copenai.summarizer.ceil_prompt(prompt)

        logger.info("history_augmented_prompt: " + history_augmented_prompt)
        return history_augmented_prompt

    def _agent_run(self, query: str, stream_callback: Callable[..., None] | None = None) -> Dict[str, Any]:
        agent = Agent(
            prompt_node=self.agent_prompt_node,
            prompt_template=agent_prompt,
            tools_manager=ToolsManager([self.web_qa_tool]),
            max_steps=4,
            streaming=False,  # Disable the native streaming callback
        )
        agent.callback_manager.on_new_token = stream_callback
        return agent.run(
            query=query,
            params={
                "Retriever": {"top_k": 3},
            },
        )
