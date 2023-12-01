import logging
from typing import Any, List, Optional
import json
from uuid import UUID

# fastapi
from fastapi import WebSocket

# langchain
from langchain.agents import AgentType, load_tools, initialize_agent
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser

# dynamic
from dynamic.classes.message import ServerMessage

CHAT_AGENTS = [
    AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
]

class DynamicAgent:
    def __init__(self, llm, agent, **kwargs):
        self.llm = llm
        self.agent = agent
        self.kwargs = kwargs

    def _initialize_agent_with_websocket(self, websocket: WebSocket):
        # TODO: Minimize the number of agent inits
        logging.info("Setting up streaming settings for agent...")
        llm = self.llm

        llm.streaming = True
        llm.verbose = True
        llm.callbacks = [WebsocketCallbackHandler(websocket)]

        # chat agent
        if self.agent in CHAT_AGENTS:
            agent_kwargs = self.kwargs.get("agent_kwargs", {})
            agent_kwargs["output_parser"] = DynamicParser()
            self.kwargs["agent_kwargs"] = agent_kwargs

        tool_list = self.kwargs.get("tool_list")
        if tool_list:
            tools = self.kwargs.get("tools", [])
            tools += load_tools(tool_list, llm=llm)
            self.kwargs["tools"] = tools

        logging.info("Initializing agent...")
        return initialize_agent(llm=llm, agent=self.agent, **self.kwargs)

class DynamicParser(ConvoOutputParser):
    def parse(self, text: str):
        if "action" not in text:
            text = json.dumps(dict(action="Final Answer", action_input=text))

        return super().parse(text)


class WebsocketCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        messsage = ServerMessage(
            content=token
        )
        await self.websocket.send_json(messsage.to_dict())