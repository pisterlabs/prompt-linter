from typing import List, Optional, Type

from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from bespokebots.services.agent.google_calendar_tools.base import GoogleCalendarBaseTool
from bespokebots.services.google_calendar import (
    GoogleCalendarEvent
)

class UpdateEventSchema(BaseModel):
    """Schema for the UpdateEventTool."""

    calendar_id: str = Field(
        ...,
        title="Calendar ID",
        description="The ID of the calendar to update the event on."
    )

    user_id: str = Field(
        ...,
        title="User ID",
        description="The user's ID, necessary for ensuring the calendar client is able to authenticate to Google."
    )

    event_id: str = Field(
        ...,
        title="Event ID",
        description="""The ID of the event to update. This field is required to ensure the correct event is being deleted from the user's calendar."""
    )


class GoogleCalendarDeleteEventTool(GoogleCalendarBaseTool):
    """Tool for deleting events in Google Calendar."""
    name: str = "delete_calendar_event"
    args_schema: Type[BaseModel] = UpdateEventSchema
    description: str = "Use this tool to delete an event from a user's calendar. You have to have the events event_id to delete it"

    def _run(
            self,
            user_id: str,
            calendar_id: str,
            event_id: str,
            run_manager: Optional[CallbackManagerForToolRun] = None, 
             ) -> GoogleCalendarEvent:
        """Delete the event from the calendar.

        Returns:
            GoogleCalendarEvent: The event that was deleted.
        """
        try:
            self.gcal_client.initialize_client(user_id=user_id)
            
            return self.gcal_client.delete_event(
                calendar_id=calendar_id,
                event_id=event_id,
            )
        except Exception as e:
            raise Exception(f"Error deleting event: {e}")
    
    def _arun(self) -> GoogleCalendarEvent:
        """Delete the event from the calendar.

        Returns:
            GoogleCalendarEvent: The event that was deleted.
        """
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")
    