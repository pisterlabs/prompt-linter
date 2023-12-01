from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

import argparse

parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument('--user_input', type=str, help='User input to add to prompt')
parser.add_argument('--schedule', action='store_true', help='Call ask_for_times')
parser.add_argument('--confirm', action='store_true', help='Call confirm_times')

args = parser.parse_args()
user_input = args.user_input

SCOPES = [
    'https://www.googleapis.com/auth/calendar', # Full control of calendars
    'https://www.googleapis.com/auth/gmail.modify' # Read, send, delete, and manage email, and manage drafts
]

CALENDAR_ID = 'alessio@decibel.vc'
OAUTH_PORT = 4567

openai.api_key = os.environ.get("OPENAI_API_KEY")
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def openai_call(prompt, temperature=0.7):
    return openai.ChatCompletion.create(
        model="gpt-4", 
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content

# This isn't good for this use case: https://twitter.com/FanaHOVA/status/1692222649920111099?s=20
# Leaving implementation here for other use cases later maybe
def anthropic_call(prompt, temperature=0.7):
    return anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=3000,
        temperature=temperature,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    ).completion

def get_calendar_events():
    creds = None
    
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    flow.authorization_url(prompt='consent', approval_prompt='force', access_type='offline')
    creds = flow.run_local_server(port=OAUTH_PORT)

    service = build('calendar', 'v3', credentials=creds)

    # List all calendars
    calendar_list = service.calendarList().list().execute().get('items', [])

    # Find the calendar ID for "alessio@decibel.vc"
    calendar_id = None
    for calendar in calendar_list:
        if calendar['id'] == CALENDAR_ID:
            calendar_id = calendar['id']
            break

    if calendar_id is None:
        print(f'Calendar {CALENDAR_ID} not found.')
        return

    # Define the time range for this week and next
    timeMin = datetime.utcnow().isoformat() + 'Z'
    timeMax = (datetime.utcnow() + timedelta(weeks=2)).isoformat() + 'Z'

    # Call the Calendar API with the specific calendar ID and time range
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=timeMin,
        timeMax=timeMax,
        maxResults=2500,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])
    if not events:
        print('No upcoming events found.')
        return

    formatted_events = map(
        lambda event: None if 'date' in event['start'] else (
            event_name := event['summary'],
            event_date := datetime.fromisoformat(event['start']['dateTime']).strftime('%Y-%m-%d'),
            event_start_time := datetime.fromisoformat(event['start']['dateTime']).strftime('%H:%M:%S'),
            event_end_time := datetime.fromisoformat(event['end']['dateTime']).strftime('%H:%M:%S'),
            event_location := 'In-Person' if ('location' in event and not event['location'].lower().startswith('http')) else 'Remote',
            f"{event_name} - {event_date}: {event_start_time} - {event_end_time} - {event_location}"
        )[-1],
        events  
    )

    formatted_events = [event for event in formatted_events if event is not None]

    return formatted_events

def ask_for_times(my_events, user_input=None):
    user_prompt = f"{user_input}\n" if user_input else ""

    prompt = f"""
    I need to respond to a scheduling email with my availability for a meeting. When suggesting times, follow these rules closely:
    - My working hours are Monday - Friday, 9am-6pm. 
    - Return the dates in this format: "Aug 14th: 12pm-3pm".
    - If an event is "In-Person", do not offer times 30 minutes before or after

    {user_prompt}
    
    These are the upcoming events for my schedule; return an email draft to respond with my availability. Do not mention any of the rules above when responding:
    
    {my_events}
    """

    print(openai_call(prompt))

def confirm_times(my_events, user_input=None):
    user_prompt = f"{user_input}\n" if user_input else ""

    prompt = f"""
    I received the following scheduling email with proposed time slots:

    {user_prompt}

    These are the upcoming events for my schedule; please confirm which of the time slots listed above would work for me. 

    {my_events}
    """

    print(openai_call(prompt))

def main():
    events = get_calendar_events()
    
    if args.schedule:
        ask_for_times(events, user_input)
    elif args.confirm:
        confirm_times(events, user_input)


if __name__ == '__main__':
    main()