import os
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
from idris.debrief import hour_mapping
from idris.utils.openai import OpenAI
from textwrap import dedent

CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/calendar.events']
gcloud_json_credentials_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
service_account = os.environ['GCLOUD_SERVICE_ACCOUNT']

class MissingDetailsException(Exception):
    pass

def format_events(event):
    time_array = event['start']['dateTime'].split('T')[1].split(':')[:2] # [hour, minute]
    return time_array

class CalendarService():
    """
    Sample Events:
        - Create: "Create an event tomorrom at 3 p.m called Sports"
                  "Create an event tomorrow from 2 p.m to 5 p.m called Sports"
    """
    def __init__(self):
        self.credentials = Credentials.from_service_account_file(gcloud_json_credentials_path, scopes=CALENDAR_SCOPES)
        self.service = build('calendar', 'v3', credentials=self.credentials, static_discovery=False)
        self.openai = OpenAI()

    def idris_calendar_brief(self):
        events = list(map(format_events, self.get_events()))

        print(events)

        if len(events) == 0:
            text = 'You have no events lined up for today.<br>'
        elif len(events) > 5:
            text = 'Looks like a busy day ahead as you have lots of events lined up.<br>'
        else:
            text = f'You have {len(events)} events lined up for today.<br>'

        if len(events) > 0:
            next_event = events[0]
            hour = int(next_event[0])
            text = f"{text} You're next event is at {hour_mapping[hour][0]}:{next_event[1]} {hour_mapping[hour][1]}."

        text += '<b>Other events:<b><br><ul>'

        for event in events[1:]:
            text += f"<li>{hour_mapping[int(event[0])][0]}:{event[1]} {hour_mapping[int(event[0])][1]}</li>"
        
        text += '</ul>'

        return text

    def get_events(self):
        now = datetime.now()
        now_iso = now.isoformat() + 'Z' # 'Z' indicates UTC time
        day_end = datetime(now.year, now.month, now.day, 23, 59, 59).isoformat() + 'Z'
        events_result = self.service.events().list(calendarId='fareedidris20@gmail.com', timeMin=now_iso, timeMax=day_end, singleEvents=True, orderBy='startTime').execute()
        events = events_result.get('items', [])
        return events

    def handle_calendar_action(self, transcript):
      prompt = dedent(f"""\
        The following is a query to create a calendar event. Identify the summary of the event, the start date and time and end date and time as well as the type of action I am doing. The default event duration is 1 hour. Default to PM unless specified otherwise. Use the CurrentDateTime for context.
        CurrentDateTime: {datetime.now().isoformat()}
        Actions: [Create, Edit, Delete]

        Structure it like this:

        Summary: summary
        Start: YYYY-MM-DDTHH-mm-ss
        End: YYYY-MM-DDTHH-mm-ss
        Type: Action
        
        {transcript}""")

      print(prompt)

      response = self.openai.create_completion(prompt)
      print(response)
      event_data = response['choices'][0]['text'].strip().split('\n')

      if len(event_data) == 4:
        summary = event_data[0].split(': ')[1]
        start_date = event_data[1].split(': ')[1]
        end_date = event_data[2].split(': ')[1]
        action = event_data[3].split(': ')[1]

        if action == 'Create':
          event = {
              'summary': summary,
              'start': {
                  'dateTime': start_date,
                  'timeZone': 'Europe/Dublin'
              },
              'end': {
                  'dateTime': end_date,
                  'timeZone': 'Europe/Dublin'
              }
          }
          print(event)
          self.service.events().insert(calendarId=os.environ['GOOGLE_CALENDAR_ID'], body=event).execute()
          print('Event Created')
        else:
          print(f'Unknown action: {action}')
      else:
        print(f'Could not create event. {event_data}')