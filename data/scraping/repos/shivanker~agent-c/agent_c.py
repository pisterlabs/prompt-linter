#!/usr/bin/env python3

import logging as log
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field

import langchain
from langchain import LLMMathChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import get_buffer_string, AgentAction, SystemMessage
from langchain.tools import StructuredTool
from langchain.utilities import (
    WikipediaAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
    GoogleSearchAPIWrapper,
)

from tools.gnews import HeadlinesTool
from tools.ytsubs import yt_transcript
from tools.reader import ReaderTool
from tools.genimg import genimg_raw, genimg_curated, img_prompt_chain

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
langchain.debug = True
# langchain.verbose = True

class SearchInput(BaseModel):
    query: str = Field(
        description="Search query",
    )
    # num_results: int = Field(
    #     default=10,
    #     description="Number of results desired. Typically this should be around 10-20 since some results contain sub-results.",
    # )


class SignalCallbackHandler(StdOutCallbackHandler):
    """Callback Handler that sends updates back to signal."""

    def __init__(self, reply_fn):
        """Initialize callback handler."""
        self.reply = reply_fn

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print(action.log)
        self.reply("⚙️🛠️ " + action.log)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        print(error)


class AgentC:
    def __init__(self, reply_fn):
        self.reply = reply_fn
        self.callback = SignalCallbackHandler(self.reply)
        self.gpt3 = ChatOpenAI(temperature=0.25, model="gpt-3.5-turbo-16k-0613")
        self.conservative_llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo-16k-0613"
        )
        self.gpt4 = ChatOpenAI(temperature=0.2, model="gpt-4-0613")
        # self.llama_chat = Replicate(
        #     model="replicate/llama70b-v2-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
        # )
        self.search_provider = GoogleSearchAPIWrapper()  # DuckDuckGoSearchAPIWrapper()
        self.search = lambda query: self.search_provider.results(query, num_results=10)
        self.wikipedia = WikipediaAPIWrapper()
        self.llm_math_chain = LLMMathChain.from_llm(
            llm=self.conservative_llm, verbose=True
        )
        self.basic_tools = [
            # Tool.from_function(
            #     name="Search",
            #     func=self.search_provider.run,
            #     description="Useful for when you need to answer questions about current events. \
            #         You can use this tool to verify your facts with latest information from the internet. \
            #         You are no longer restricted by your out-of-date training data. \
            #         You should ask targeted questions. \
            #         When you can't figure out what to do with a message, try searching for the keywords using this tool. \
            #         If you can't find what you were looking for in the results of this tool, \
            #         DO NOT invent information. Just say \"I couldn't find it on the internet.\".",
            # ),
            Tool(
                name="Calculator",
                func=self.llm_math_chain.run,
                description="Useful for when you need to answer questions about math or perform mathematical operations.",
            ),
            Tool(
                name="Wikipedia",
                func=self.wikipedia.run,
                description="Useful for when you need to look up facts like from an encyclopedia. \
                    Remember, this is a high-quality trusted source.",
            ),
        ]
        self.tools = (
            self.basic_tools
            + [
                StructuredTool.from_function(
                    name="Search",
                    func=self.search,
                    description="Useful for when you need to answer questions about current events. \
                        You can use this tool to verify your facts with latest information from the internet. \
                        You are no longer restricted by your out-of-date training data. \
                        You should ask targeted questions. \
                        When you can't figure out what to do with a message, try searching for the keywords using this tool. \
                        If you can't find what you were looking for in the results of this tool, \
                        DO NOT invent information. Just say \"I couldn't find it on the internet.\".",
                    args_schema=SearchInput,
                ),
                Tool.from_function(
                    name="ImageGenerator",
                    func=(lambda p: genimg_curated(p, self.reply)),
                    description="Useful when you need to create an image that the user asks you to. \
                    This tool returns a URL of a human-visible image based on text keywords. You \
                    can return this URL when the user asks for an image. In the input you need to \
                    describe a scene in English language, mostly using keywords should be okay \
                    though.",
                ),
                HeadlinesTool(),
                ReaderTool(),
                Tool(
                    name="YoutubeTranscriptFetcher",
                    func=yt_transcript,
                    description="Useful for when you need to fetch the transcript of a YouTube video \
                        to understand it better, find something in it, or to explain it to the user.",
                ),
            ]
            # + load_tools(["open-meteo-api"], llm=self.conservative_llm)
        )
        self.memory_key = "chat_history"
        self.memory = ConversationBufferWindowMemory(
            k=20, memory_key=self.memory_key, return_messages=True
        )  # return messages is always true in "Window" memory - it's designed for chat agents
        openai_kwargs = {
            "extra_prompt_messages": [
                MessagesPlaceholder(variable_name=self.memory_key)
            ],
            "system_message": SystemMessage(
                content="Your name is SushiBot. You are a helpful AI assistant. Keep the \
                conversation natuarl and flowing, don't respond with closing statements like \
                'Is there anything else?'. If you don't know something, look it up on the \
                internet. If Search results are not useful, try to navigate to known expert \
                websites to fetch real, up-to-date data, and then root your answers to those facts."
            ),
        }
        self.gpt4_single = initialize_agent(
            self.tools,
            self.gpt4,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            agent_kwargs=openai_kwargs,
            callbacks=[self.callback],
        )
        self.gpt4_multi = initialize_agent(
            self.tools,
            self.gpt4,
            agent=AgentType.OPENAI_MULTI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            agent_kwargs=openai_kwargs,
            callbacks=[self.callback],
        )
        self.gpt3_single = initialize_agent(
            self.tools,
            self.gpt3,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            agent_kwargs=openai_kwargs,
            callbacks=[self.callback],
        )
        chat_history = MessagesPlaceholder(variable_name=self.memory_key)
        self.react = initialize_agent(
            self.tools,
            self.gpt4,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            agent_kwargs={
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", self.memory_key],
            },
            callbacks=[self.callback],
        )
        # TODO: Configure temperature, tools
        # Next steps: must use chat model, base model needs very structured prompts, not ideal for signal
        # Need to implement a BaseChatModel::_generate on top of Replicate (which is only an LLM)
        # But also, replicate doesn't expose an api which takes message hisory or roles ....
        # so would probably have to craft a system prompt to make it chat-worthy
        # or look into LLama-API
        # self.llama = initialize_agent(
        #     [],
        #     self.llama_chat,
        #     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        #     memory=self.memory,
        #     verbose=True,
        # )
        # TODO: Try PlanAndExecute agents
        self.agent = self.gpt4_multi

    def handle(self, msg):
        self.reply(self.handle2(msg))

    def handle2(self, msg):
        if msg == "/reset":
            self.memory.clear()
            return "Your session has been reset."
        elif msg == "/gpt4":
            self.agent = self.gpt4_single
            return "You're now chatting to GPT4 Functions model (single)."
        elif msg == "/multi":
            self.agent = self.gpt4_multi
            return "You're now chatting to GPT4 Functions model (multi)."
        elif msg == "/gpt3":
            self.agent = self.gpt3_single
            return "You're now chatting to GPT3.5."
        elif msg == "/react":
            self.agent = self.react
            return "You're now chatting to the ReAct model."
        # elif msg == "/llama":
        #     self.agent = self.llama
        #     return "You're now chatting to the LLama-v2-70B model."
        elif msg == "/memory":
            history = get_buffer_string(
                self.memory.buffer,
                human_prefix=self.memory.human_prefix,
                ai_prefix=self.memory.ai_prefix,
            )
            return history if history else "Memory is empty."
        elif msg.startswith("/genimg"):
            return genimg_raw(msg[8:])
        elif msg.startswith("/curated"):
            return genimg_curated(msg[9:])
        elif msg.startswith("/imgprompt"):
            return img_prompt_chain(msg[len("/imgprompt") + 1 :])["text"]
        return self.agent.run(msg)
