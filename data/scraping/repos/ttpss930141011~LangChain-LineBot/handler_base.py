from abc import ABC, abstractmethod
from typing import List, Type

from langchain.agents import AgentExecutor
from linebot.v3.messaging.models.message import Message

from src.interactor.dtos.event_dto import EventInputDto
from src.interactor.interfaces.repositories.agent_executor_repository import (
    AgentExecutorRepositoryInterface,
)


class Handler(ABC):
    def __init__(self, successor: Type["Handler"] = None):
        self._successor = successor

    def _get_agent_executor(
        self,
        input_dto: EventInputDto,
        repository: AgentExecutorRepositoryInterface,
    ) -> AgentExecutor:
        """
        Retrieves the agent executor associated with the current window.

        :param None: This function does not take any parameters.
        :return: None
        """

        window_id = input_dto.window.get("window_id")

        agent_executor = repository.get(
            window_id=window_id,
        )
        if agent_executor is None:
            agent_executor = repository.create(
                window_id=window_id,
            )
        return agent_executor

    @abstractmethod
    def handle(
        self,
        input_dto: EventInputDto,
        repository: AgentExecutorRepositoryInterface,
        response: List[Message],
    ) -> List[Message]:
        pass
