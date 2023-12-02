from .models import BasePool
from .langChainAgent import LangChainAgent
from datetime import datetime, date, time
from enum import Enum
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

from langchain.output_parsers import ListOutputParser

from pydantic import BaseModel, Field, field_validator
from typing import Coroutine, Optional, Type, Any
from uuid import UUID
import requests


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
        - The user has no high intention to book or search ticket. For example: '不訂了', '不查了', '不想要'
        - The user want to break or exit this conversation.
        - The topic of the conversation isn't highly related to HSR(高鐵).
        """

        def _run(self, *args: Any, **kwargs: Any) -> str:
            set_exit_state(user_id)
            return "The process exited successfully."
    return ExitTool()


class Station(str, Enum):
    nangang = "南港"
    taipei = "台北"
    banqiao = "板橋"
    taoyuan = "桃園"
    hsinchu = "新竹"
    miaoli = "苗栗"
    taichung = "台中"
    changhua = "彰化"
    yunlin = "雲林"
    chiayi = "嘉義"
    tainan = "台南"
    zuoying = "左營"
    other = None


class HsrSearchInput(BaseModel):
    departure_date: date = Field(
        description="Departure date format as iso8601 which only contains date.")
    departure_time: time = Field(
        description="Exact departure time from user format as iso8601 which only contains time.")
    station_from: Station = Field(description="Departure station from user.")
    station_to: Station = Field(description="Destination station from user.")
    adult_count: int = Field(
        description="Numbers of aldult to take HSR from user.")
    student_count: int = Field(
        description="Numbers of students to take HSR from user.")

    @field_validator("station_from", "station_to", mode="before")
    def station_validator(cls, value):
        if value not in list(map(lambda x: x.value, Station)):
            value = Station.other
        return value


def getHsrSearchTool(user_id: str):
    class HsrSearchTool(BaseTool):
        name = "HSRSearchTool"
        # description = f"Useful to get the HSR(高鐵) timetable. You have to ask the user '你想要搭乘的日期和時間' and '你的出發站和抵達站'. Also, you will get a session_id for booking HSR(高鐵) ticket, it is pretty important. Current time is {datetime.now().strftime('%c')}"
        description = f"Useful to get the HSR(高鐵) timetable. You have to ask the user '你想要搭乘的日期和時間', '你的出發站和抵達站', and '成人和學生票的數量', and then you will get a timetable for booking HSR(高鐵) ticket. Current time is {datetime.now().strftime('%c')}"
        args_schema: Optional[Type[BaseModel]] = HsrSearchInput

        def _run(self, departure_date: date, departure_time: time, station_from: Station, station_to: Station, adult_count: int, student_count: int) -> str:
            if station_from == Station.other or station_to == Station.other:
                return f"Error! The `station_from` or `station_to` is not found, it should be in {list(map(lambda x: x.value, Station))}"

            request = requests.get(
                "https://api.squidspirit.com/hsr/search",
                json={
                    "station_from": station_from.value,
                    "station_to": station_to.value,
                    "adult_count": adult_count,
                    "child_count": 0,
                    "heart_count": 0,
                    "elder_count": 0,
                    "student_count": student_count,
                    "departure": datetime.combine(departure_date, departure_time).isoformat()
                }
            )

            if request.status_code != 200:
                return request.json()

            get_agent_pool_instance().set_session_id(
                user_id, request.json()['session_id'])
            # return f"The session_id is '{request.json()['session_id']}' (You must remember it); the timetable is {request.json()['data']}"
            return f"The timetable is {request.json()['data']}"

        def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
            raise Exception()
    return HsrSearchTool()


class HsrBookInput(BaseModel):
    index: int = Field(
        description="The index (start from 0) of desiring train no. from the timetable.")
    # session_id: UUID = Field(
    #     description="session_id format as UUID you just got from `HSRSearchTool`")


def getHsrBookTool(user_id: str):
    class HsrBookTool(BaseTool):
        name = "HSRBookTool"
        # description = "Useful to book HSR(高鐵) ticket. If you don't konw the timetable and session_id, use `HSRSearchTool` first."
        description = "Useful to book HSR(高鐵) ticket, and it will return a booking information screenshot url. If you don't konw the timetable, use `HSRSearchTool` first."
        args_schema: Optional[Type[BaseModel]] = HsrBookInput

        # def _run(self, index: int, session_id: UUID) -> Any:
        def _run(self, index: int) -> Any:
            session_id = get_agent_pool_instance().get_session_id(user_id)
            from backenddb.appModel import find_hsr_data
            hsr_data, founded = find_hsr_data(user_id)
            if not founded:
                set_exit_state(user_id)
                return "User hasn't set hsr booking data yet, ask user to fill in required message"
            request = requests.post(
                f"https://api.squidspirit.com/hsr/book/{session_id}",
                json={
                    "selected_index": index,
                    "id_card_number": hsr_data.id_card_number,
                    "phone": hsr_data.phone_number,
                    "email": hsr_data.email,
                    "debug": True,
                }
            )

            if request.status_code != 200:
                return request.json()
            set_exit_state(user_id)
            return f"The booking information screenshot url is {request.json()['data']}, the user can go to pay bill for it."

        def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
            raise Exception()
    return HsrBookTool()


class HsrAgentPool(BasePool):
    def __init__(self) -> None:
        super().__init__()
        self.sessions: dict[str, UUID] = {}

    def add(self, user_id: str) -> LangChainAgent:
        agent = self.pool[user_id] = LangChainAgent(
            tools=[
                getExitTool(user_id),
                getHsrSearchTool(user_id),
                getHsrBookTool(user_id)
            ],
            memory=ConversationBufferMemory(
                memory_key="hsr", return_messages=True),
            timeout=-1
        )
        return agent

    def get_session_id(self, user_id: str) -> UUID:
        return self.sessions.get(user_id)

    def set_session_id(self, user_id: str, session_id: UUID) -> None:
        self.sessions[user_id] = session_id


__hsr_agent_pool_instance = HsrAgentPool()


def get_agent_pool_instance():
    return __hsr_agent_pool_instance
