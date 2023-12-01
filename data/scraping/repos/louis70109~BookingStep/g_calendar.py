from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import datetime
import urllib


class GoogleCalendarGeneratorInput(BaseModel):
    """Input for Google Calendar Generator."""

    dates: str = Field(
        ...,
        description=f"Datetime symbol if text contained. format should be 'YYYYMMDDTHHMMSS/YYYYMMDDTHHMMSS'. Current time is {datetime.date.today()}")
    title: str = Field(
        ...,
        description="Calendar Title symbol for reserve schedule.")
    description: str = Field(
        ...,
        description="Calendar Summary text symbol for schedule description.")
    location: str = Field(
        ...,
        description="Calendar location symbol for reservation.")


class CalendarTool(BaseTool):
    name = "google_calendar_reservation"
    description = f"""
Generate Google Calendar url from user text first when containing time, date.

"""

    @staticmethod
    def create_gcal_url(
            title='看到這個..請重生',
            date='20230524T180000/20230524T220000',
            location='那邊',
            description=''):
        base_url = "https://www.google.com/calendar/render?action=TEMPLATE"
        event_url = f"{base_url}&text={urllib.parse.quote(title)}&dates={date}&location={urllib.parse.quote(location)}&details={urllib.parse.quote(description)}"
        return event_url+"&openExternalBrowser=1"

    def _run(self, dates: str, title: str, description: str, location: str):
        print('Google Calendar')
        print('時間：'+dates)
        print('標題：'+title)
        print('描述：'+description)
        print('地點：'+location)
        result = self.create_gcal_url(title, dates, location, description)

        return result

    args_schema: Optional[Type[BaseModel]] = GoogleCalendarGeneratorInput
