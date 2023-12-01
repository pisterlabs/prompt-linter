import os

from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import json
import os
import pickle
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
import dateparser, re

SCOPES = ['https://www.googleapis.com/auth/calendar']

def parse_and_create_event(input_str, service):
    # Extract information from the input
    match = re.search(r'(\d+)(?:am|pm)', input_str)
    time_str = match.group(0) if match else ''
    date_str = 'tomorrow' if 'tomorrow' in input_str else ''
    duration_match = re.search(r'(\d+)\s?(?:hour|hr)', input_str)
    duration_str = duration_match.group(0) if duration_match else '1 hour'

    # Parse the date and time
    start_time = dateparser.parse(f'{date_str} {time_str}')
    duration = dateparser.parse(f'{duration_str}').time()
    end_time = start_time + datetime.timedelta(hours=duration.hour)

    # Create the event
    summary = 'New Event'
    location = ''
    timezone = 'America/New_York'
    create_event(service, summary, location, start_time.isoformat(), end_time.isoformat(), timezone)

def get_credentials():
    """Load or create Google API credentials."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def create_event(service, summary, location, start_time, end_time, timezone='America/New_York'):
    """Create a new event in Google Calendar."""
    event = {
        'summary': summary,
        'location': location,
        'start': {
            'dateTime': start_time,
            'timeZone': timezone,
        },
        'end': {
            'dateTime': end_time,
            'timeZone': timezone,
        },
    }

    event = service.events().insert(calendarId='primary', body=event).execute()
    print(f'Event created: {event.get("htmlLink")}')
    return event

def list_events(service, **kwargs):
    num_events=10

    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=num_events, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])
    return events

class GoogleCalendarAgent(BaseMultiActionAgent):
    @property
    def input_keys(self):
        return ["input"]

    # Synchronous plan method to decide which action to take based on the input
    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        # If input contains "create an event", plan to create an event
        if "create an event" in kwargs["input"].lower():
            return [AgentAction(tool="GoogleCalendarCreateEvent", tool_input=kwargs["input"], log="")]
        else:
            return [AgentAction(tool="GoogleCalendarListEvents", tool_input=kwargs["input"], log="")]

    # Asynchronous plan method to decide which action to take based on the input
    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        # If input contains "create an event", plan to create an event
        if "create an event" in kwargs["input"].lower():
            return [AgentAction(tool="GoogleCalendarCreateEvent", tool_input=kwargs["input"], log="")]
        else:
            return [AgentAction(tool="GoogleCalendarListEvents", tool_input=kwargs["input"], log="")]


if __name__ == "__main__":
    creds = get_credentials()
    service = build('calendar', 'v3', credentials=creds)

    tools = [
        Tool(
            name="GoogleCalendarCreateEvent",
            func=lambda tool_input: parse_and_create_event(tool_input, service),
            description="Create a new event in Google Calendar"
        ),
        Tool(
            name="GoogleCalendarListEvents",
            func=lambda _: list_events(service),  # Accept an argument but don't use it
            description="List upcoming events in Google Calendar"
        ),
    ]

    llm=ChatOpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=False)

    while True:
        print('\n\n\033[31m' + 'Ask a question' + '\033[m')
        user_input = input()
        print('\033[31m' + str(agent_chain.run(input=user_input, chat_history=[]) + '\033[m'))
