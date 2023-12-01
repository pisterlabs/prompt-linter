from ast import literal_eval
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from datetime import datetime
from urllib import parse
    
class GoogleCalendarGeneratorInput(BaseModel):
    """Input for Google Calendar Generator."""

    dates: str = Field(
        ...,
        description="Datetime symbol for google calendar url."
    )
    title: str = Field(
        ...,
        description="Title symbol for google calendar url."
    )
    description: str = Field(
        ...,
        description="Summary text symbol for google calendar url."
    )
    location: str = Field(
        ...,
        description="Calendar location symbol for google calendar."
    )

class CalendarTool(BaseTool):
    name = "create_google_calendar_url"
    description = f"""
    Generate Google Calendar API url from CalendarTextSplit text.
    Dates format should be like 'YYYYMMDDTHHMMSS/YYYYMMDDTHHMMSS'.
    Cuurent time {datetime.now()}.
    """

    def _run(self, title: str, dates: str, description: str, location: str):
        print('標題:', title)
        print('時間:', dates)
        print('描述:', description)
        print('地點:', location)
        result = self.create_gcal_url(title, dates, location, description)

        return result
    
    def create_gcal_url(
            self,
            title="梅竹黑客松",
            dates='20231019T180000/20231019T220000',
            location='國立清華大學',
            description="為期兩日的Coding黑客松"):
        base_url = "https://www.google.com/calendar/render?action=TEMPLATE"
        event_url = f"{base_url}&text={parse.quote(title)}&dates={dates}&location={parse.quote(location)}&details={parse.quote(description)}"
        
        return event_url + "&openExternalBrowser=1"

    args_schema: Optional[Type[BaseModel]] = GoogleCalendarGeneratorInput