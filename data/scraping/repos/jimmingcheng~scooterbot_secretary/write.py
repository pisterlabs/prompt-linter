import arrow
import json
import openai
from typing import Tuple

from secretary.calendar import event_start_time
from secretary.calendar import get_calendar_service
from secretary.database import UserTable
from secretary.todo_emailer import send_email
from secretary.todo_emailer import should_remind_today


def add_todo_from_prompt(user_id: str, user_prompt: str) -> str:
    instructions_prompt = f"""
Current time is {arrow.now()}

The user will describe a task in plain English. Convert it to this format:
{{"description": "...", "date": "YYYY-MM-DD", "confirmation_message": "..."}}

Here's an example:
{{"description": "Cut my hair", "date": "2021-03-02", "confirmation_message": "I'll remind you to cut your hair next Tuesday, March 2"}}

Output only the json.
    """

    completion_text = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': instructions_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    ).choices[0].message.content

    resp = json.loads(completion_text)

    todo, reminder_days_before = add_todo(user_id, resp['description'], arrow.get(resp['date']))

    response_speech = resp['confirmation_message']

    if reminder_days_before > 0:
        todo_start_time = event_start_time(todo)

        duration = todo_start_time.humanize(
            other=todo_start_time.shift(days=-reminder_days_before),
            only_distance=True
        )
        response_speech += f" I'll remind you {duration} before."

    return response_speech


def add_todo(user_id: str, description: str, date: arrow.Arrow) -> Tuple[dict, int]:
    user = UserTable().get(user_id)
    google_apis_user_id = user['google_apis_user_id']
    calendar_id = user['todo_calendar_id']

    cal_service = get_calendar_service(google_apis_user_id)

    event = cal_service.events().insert(
        calendarId=calendar_id,
        body={
            'summary': description,
            'start': {'date': date.format('YYYY-MM-DD')},
            'end': {'date': date.format('YYYY-MM-DD')},
        },
    ).execute()

    reminder_days_before = auto_set_reminder(event)

    event = cal_service.events().update(
        calendarId=calendar_id,
        eventId=event['id'],
        body=event,
    ).execute()

    if should_remind_today(event, []):
        send_email(google_apis_user_id, event)

    return event, reminder_days_before


def get_todo_reminder_days_before(date: arrow.Arrow) -> int:
    delta = date - arrow.now()
    if delta.days > 28:
        return 7
    elif delta.days > 5:
        return 2
    else:
        return 0


def convert_to_all_day_event(event: dict) -> None:
    # all todos are represented as all-day events, so convert if necessary
    event['start']['date'] = arrow.get(event['start']['dateTime']).format('YYYY-MM-DD')
    event['end']['date'] = event['start']['date']
    del event['start']['dateTime']
    del event['end']['dateTime']


def auto_set_reminder(event: dict) -> int:
    event['reminders']['useDefault'] = False

    if not event['reminders'].get('overrides'):
        event['reminders']['overrides'] = []

    reminders = event['reminders']['overrides']

    # reminders are represented as popup reminders
    popup_reminder = None
    for reminder in reminders:
        if reminder['method'] == 'popup':
            popup_reminder = reminder

    if not popup_reminder:
        popup_reminder = {'method': 'popup'}
        reminders.append(popup_reminder)

    reminder_days_before = get_todo_reminder_days_before(event_start_time(event))
    for reminder in reminders:
        # this will update all reminders for this event, but only the popup
        # reminder is used for email reminder purposes
        reminder['minutes'] = reminder_days_before * 1440

    return reminder_days_before
