from typing import List, Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from bespokebots.services.agent.google_calendar_tools.base import GoogleCalendarBaseTool


class ViewEventsSchema(BaseModel):
    """Schema for the ViewEventsTool."""

    calendar_id: str = Field(
        ...,
        title="Calendar ID",
        description="The ID of the calendar to view events from.",
    )

    user_id: str = Field(
        ...,
        title="User ID",
        description="The user's ID, necessary for ensuring the calendar client is able to authenticate to Google."
    )

    time_min: Optional[str] = Field(
        None,
        title="Time Min",
        description="The start time of the time range to view events from. In ISO 8601 format with timezone offest.",
    )
    time_max: Optional[str] = Field(
        None,
        title="Time Max",
        description="The end time of the time range to view events from. In ISO 8601 format with timezone offest.",
    )


class GoogleCalendarViewEventsTool(GoogleCalendarBaseTool):
    """Tool for viewing events in Google Calendar."""

    name: str = "view_calendar_events"
    description: str = """Use this tool to help you answer questions a human may have about their schedule. Think of this as the your calendar search tool. The human may ask you to search for events over any timeframe but the default is one week in the future. The default timezone is America/New_York."""

    args_schema: Type[ViewEventsSchema] = ViewEventsSchema

    def _run(
        self,
        calendar_id: str,
        user_id: str,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """View events in Google Calendar."""
        try:
            #ensure the gcal_client has been authenticated
            self.gcal_client.initialize_client(user_id)

            gcal_events = self.gcal_client.get_calendar_events(
                calendar_id, time_min, time_max
            )
            if not gcal_events:
                return {"message": "No events found."}
            else:
                events = [event.to_dict() for event in gcal_events]
                return events
        except Exception as e:
            raise Exception(f"An error occurred: {e}")

    async def _arun(self, calendar_id: str, time_min: str, time_max: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> dict:
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")
