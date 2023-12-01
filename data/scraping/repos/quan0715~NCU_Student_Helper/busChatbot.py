from .models import BasePool
from .langChainAgent import LangChainAgent
from enum import Enum
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import EnumOutputParser
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Type, Any
from uuid import uuid4
from .bus_api.api import BusAPI
from .bus_api.stop_data import StopData


def set_exit_state(user_id: str) -> None:
    from .chatBotModel import default_message
    from .chatBotExtension import jump_to
    jump_to(default_message, user_id, True)
    get_agent_pool_instance().remove(user_id)


def getExitTool(user_id: str):
    class ExitTool(BaseTool):
        name = "ExitTool"
        # description = "Useful when user want to exit or break this conversation in any situation."
        description = """
        Useful in the situations below:
        - The user has no high intention to book or search ticket. For example: '不查了', '不想要'
        - The user want to break or exit this conversation.
        - The topic of the conversation isn't highly related to bus.
        """

        def _run(self, *args: Any, **kwargs: Any) -> str:
            set_exit_state(user_id)
            return "The process exited successfully."
    return ExitTool()


class BusLine(str, Enum):
    line132 = "132"
    line133 = "133"
    line172 = "172"
    line173 = "173"
    other = None


class BusInfoInput(BaseModel):
    direction: int = Field(
        description="If the bus is going to NCU(中央大學 or 中央), it should be 1; if the bus is leaving from NCU(中央大學 or 中央), it should be 0.")
    destination: int = Field(
        description="Whether the bus is going to (leaving from) 火車站 (1) or 高鐵站 (0).")

    @field_validator("line", mode="before")
    def station_validator(cls, value):
        if value not in list(map(lambda x: x.value, BusLine)):
            value = BusLine.other
        return value


class BusInfoTool(BaseTool):
    name = "BusInfoTool"
    description = "Useful to get when the bus will arrive the bus stop."

    args_schema: Optional[Type[BaseModel]] = BusInfoInput

    def _run(self, direction: int, destination: str) -> str:

        stop = "中央大學正門"
        if destination == 1:
            line_coll = ["172", "173"]
            stop_list = list(BusAPI.get_all_stops("172"))
            if direction == 1:
                stop = "中壢火車站"
        else:
            line_coll = ["132", "133"]
            stop_list = list(BusAPI.get_all_stops("132"))
            if direction == 1:
                stop = "高鐵桃園站"

        print(stop_list)

        stop_info_coll = list(
            map(lambda x: BusAPI.get_bus_data(x, direction), line_coll))
        result_coll: list[StopData] = []
        for infos in stop_info_coll:
            for info in infos:
                if info.stop_name == stop:
                    result_coll.append(info)
        sorted(result_coll, key=lambda x: x.next_bus_time)
        result = result_coll[0]
        return f"[System] The next bus is arriving at {result.next_bus_time} and the status is {result.stop_status} at Station {result.stop_name}"


class BusAgentPool(BasePool):
    def add(self, user_id: str) -> LangChainAgent:
        agent = self.pool[user_id] = LangChainAgent(
            tools=[getExitTool(user_id), BusInfoTool()],
            memory=ConversationBufferMemory(
                memory_key="bus", return_messages=True),
        )
        return agent


__bus_agent_pool_instance = BusAgentPool()


def get_agent_pool_instance():
    return __bus_agent_pool_instance
