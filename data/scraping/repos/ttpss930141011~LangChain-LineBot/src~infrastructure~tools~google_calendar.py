import datetime
import urllib
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


def create_gcal_url(
    title="What?", date="20230524T180000/20230524T220000", location="Where?", description=""
):
    """
    Generate a Google Calendar URL for creating a new event.

    Args:
        title (str): The title of the event. Defaults to "What?".
        date (str): The date and time of the event in the format "yyyyMMddTHHmmss/yyyyMMddTHHmmss".
                    Defaults to "20230524T180000/20230524T220000".
        location (str): The location of the event. Defaults to "Where?".
        description (str): The description of the event. Defaults to an empty string.

    Returns:
        str: The URL for creating a new event in Google Calendar.

    """

    base_url = "https://www.google.com/calendar/render?action=TEMPLATE"
    event_url = f"{base_url}&text={urllib.parse.quote(title)}&dates={date}\
            &location={urllib.parse.quote(location)}&details={urllib.parse.quote(description)}"

    return event_url + "&openExternalBrowser=1"


class GoogleCalendarGeneratorInput(BaseModel):
    """Input for Google Calendar Generator."""

    dates: str = Field(
        ...,
        description=f"Datetime symbol if text contained. format should be \
        'YYYYMMDDTHHMMSS/YYYYMMDDTHHMMSS'. Current time is {datetime.date.today()}",
    )
    title: str = Field(..., description="Calendar Title symbol for reserve schedule.")
    description: str = Field(
        ..., description="Calendar Summary text symbol for schedule description."
    )
    location: str = Field(..., description="Calendar location symbol for reservation.")


class GoogleCalendarTool(BaseTool):
    name = "google_calendar_reservation"
    description = "Generate Google Calendar url from user text first when containing time, date."
    args_schema: Optional[Type[BaseModel]] = GoogleCalendarGeneratorInput

    def _run(self, dates: str, title: str, description: str, location: str):
        """
        Generates a Google Calendar URL based on the given parameters.

        Args:
            dates (str): The dates of the event.
            title (str): The title of the event.
            description (str): The description of the event.
            location (str): The location of the event.

        Returns:
            str: The generated Google Calendar URL.
        """

        result = create_gcal_url(title, dates, location, description)

        return result
