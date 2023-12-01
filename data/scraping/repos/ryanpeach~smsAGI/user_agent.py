from pathlib import Path
from typing import List

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from lib.agents.task_prioritization_agent import TaskPrioritizationAgent
from lib.config.prompts import Prompts
from lib.config.tools import BaseTool, Tools
from lib.sql import Goal, SuperAgent, TaskListItem, ThreadItem


class UserAgent:
    def __init__(self, super_agent: SuperAgent, session: Session, config: Path):
        self.tools = self.get_tools(
            config=config, session=session, super_agent=super_agent
        )
        self.llm = self.get_llm(config=config)
        self.user_agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        self.super_agent = super_agent
        self.session = session
        goals = Goal.get_goals(session=self.session, agent=self.super_agent)
        prompts = Prompts(config=config)
        self.objective = Goal.get_prompt(goals=goals, prompts=prompts)
        self.task_prioritization_agent = TaskPrioritizationAgent(
            super_agent=super_agent, session=session, config=config
        )

    @staticmethod
    def get_llm(config: Path) -> ChatOpenAI:
        """Gets the LLM from the config file."""
        return Tools(config=config).get_zero_shot_llm()

    @staticmethod
    def get_tools(
        config: Path, session: Session, super_agent: SuperAgent
    ) -> list[BaseTool]:
        """Gets a list of tools from the config file."""
        tools = Tools(config=config)
        tools_list = [
            tools.get_todo_tool(),
            tools.get_send_message_tool(session=session, super_agent=super_agent),
            tools.get_send_message_wait_tool(session=session, super_agent=super_agent),
        ]
        return [tool for tool in tools_list if tool is not None]

    async def arun(self, user_msg: str):
        # Get past messages from the database and assemble a message list
        thread = ThreadItem.get_all(session=self.session, super_agent=self.super_agent)

        def assemble_messages() -> List[BaseMessage]:
            messages: List[BaseMessage] = [
                SystemMessage(content=self.objective),
                *thread,
                HumanMessage(content=user_msg),
            ]
            return messages

        messages = assemble_messages()
        num_tokens = self.llm.get_num_tokens_from_messages(messages=messages)
        if self.llm.max_tokens is not None:
            while num_tokens > self.llm.max_tokens:
                if len(thread) == 0:
                    raise ValueError("Thread is empty, but still too many tokens.")
                thread.pop(0)
                messages = assemble_messages()
                num_tokens = self.llm.get_num_tokens_from_messages(messages=messages)

        # Run the LLM
        task_str = await self.user_agent.arun(messages)

        # Create a new task list item
        task_list_item = TaskListItem(
            description=task_str,
            super_agent=self.super_agent,
            priority=0.5,
        )

        # Prioritize the task
        await self.task_prioritization_agent.arun(task_list_item=task_list_item)

        # Add the task to the database
        self.session.add(task_list_item)
