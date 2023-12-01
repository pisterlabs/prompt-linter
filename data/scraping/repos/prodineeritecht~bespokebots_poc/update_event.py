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

    summary: Optional[str] = Field(
        None,
        title="Summary",
        description="""The name of the event, found in the summary field. This is a good field to search on, 
        but is also good for confirming we are updating the right event. If event id is not provided, this field is required"""
    )

    event_id: Optional[str] = Field(
        None,
        title="Event ID",
        description="""The ID of the event to update. This is the preferred way to identify an event on a user's calendar.
        If event id is available, then pass that in along with any other information."""
    )
    
    start_time: Optional[str] = Field(
        None,
        title="Start Time",
        description="The start date and time of the event. In ISO 8601 format. If event_id is not provided, this is required"
    )
    end_time: Optional[str] = Field(
        None,
        title="End Time",
        description="The end date and time of the event. In ISO 8601 format. if event_id is not provided, this is required"
    )

    update_fields: dict = Field(
        ...,
        title="Update Fields",
        description="""A dictionary of fields to update. The keys are the fields to update, and the values are the new values."""
    )



class GoogleCalendarUpdateEventTool(GoogleCalendarBaseTool):
    """Tool for updating events in Google Calendar."""

    name: str = "update_calendar_event"
    description: str = """Use this tool to help you update events in Google Calendar. 
This tool will use either the event_id, or the summary, start_time and end_time to find the event to update. Once
it finds an event, it will update the event with the update_fields by setting the key in the calendar event to the value from update fields.
If the event cannot be found, an error is raised with information about what went wrong. 
"""
# Below is an example of the updateable fields in a Google Calendar event JSON. Use this to help figure out how to construct the 'update_fields' parameter:
# {"summary": "Rob Whiteston and Jessica Loy","description": "Event Name: Recruiter Screen","location": "Google Meet (instructions in description)","start": {"dateTime": "2023-05-19T15:00:00-04:00","timeZone": "America/Chicago"},"end": {"dateTime": "2023-05-19T15:30:00-04:00","timeZone": "America/Chicago"}}

    args_schema: Type[UpdateEventSchema] = UpdateEventSchema

    def _run(
        self,
        calendar_id: str,
        user_id: str,
        summary: Optional[str] = None,
        event_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        update_fields: dict = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """Update an event in Google Calendar."""
        try:
            #ensure the gcal_client has been authenticated
            self.gcal_client.initialize_client(user_id)

            if event_id is None and summary is None and start_time is None and end_time is None:
                raise Exception("Either an event id or start, end and summary are required to find and then update and event")
            
            if event_id: 
                #look up the event from gcal and then update it with the update_fields  
                gcal_event = self.gcal_client.get_event(calendar_id, event_id)  

                if not gcal_event:
                    raise Exception("Unable to find an event with eventId={event_id} on calendarId={calendar_id}")
            else:
                #look up the event from gcal and then update it with the update_fields  
                
                events = self.gcal_client.get_calendar_events(calendar_id, start_time, end_time)  
                found_event = [event for event in events if event.summary == summary][0]
                if not found_event:
                    raise Exception("Unable to find an event with summary={summary} on calendarId={calendar_id} between {start_time} and {end_time}")
                
                gcal_event = self.gcal_client.get_event(calendar_id, found_event.event_id)
                

            #update the gcal_event with the update_fields
            for key, value in update_fields.items():
                gcal_event[key] = value
            
            
            updated_event = self.gcal_client.update_event(
                calendar_id, gcal_event
            )
            return updated_event
        except Exception as e:
            raise Exception(f"An error occurred: {e}")

    async def _arun(self, calendar_id: str, summary: str, event_id: str, start_time: str, end_time: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> dict:
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")