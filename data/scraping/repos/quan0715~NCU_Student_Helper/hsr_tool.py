from datetime import datetime, date, time
from enum import Enum
from langchain.output_parsers import PydanticOutputParser, EnumOutputParser
from langchain.schema import OutputParserException
from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from pydantic import BaseModel, Field, field_validator
from typing import Coroutine, Dict, Optional, Type, Any, Union
from uuid import UUID
import requests


# 因為 AI 太笨，記不住伺服器回傳的 session id，
# 可能要另外用一個 dict[line_id, session_id] 紀錄
session_id: UUID


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

        global session_id
        session_id = request.json()['session_id']
        # return f"The session_id is '{request.json()['session_id']}' (You must remember it); the timetable is {request.json()['data']}"
        return f"The timetable is {request.json()['data']}"

    def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        raise Exception()


class HsrBookInput(BaseModel):
    index: int = Field(
        description="The index (start from 0) of desiring train no. from the timetable.")
    # session_id: UUID = Field(
    #     description="session_id format as UUID you just got from `HSRSearchTool`")


class HsrBookTool(BaseTool):
    name = "HSRBookTool"
    # description = "Useful to book HSR(高鐵) ticket. If you don't konw the timetable and session_id, use `HSRSearchTool` first."
    description = "Useful to book HSR(高鐵) ticket, and it will return a booking information screenshot url. If you don't konw the timetable, use `HSRSearchTool` first."
    args_schema: Optional[Type[BaseModel]] = HsrBookInput

    # def _run(self, index: int, session_id: UUID) -> Any:
    def _run(self, index: int) -> Any:

        global session_id
        request = requests.post(
            f"https://api.squidspirit.com/hsr/book/{session_id}",
            json={
                "selected_index": index,
                "id_card_number": "id_card_number",
                "phone": "phone",
                "email": "email",
                "debug": True,
            }
        )

        if request.status_code != 200:
            return request.json()
        return f"The booking information screenshot url is {request.json()['data']}, the user can go to pay bill for it."

    def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        raise Exception()
