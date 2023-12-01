import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, TypedDict

import openai
import pytz
from dataclasses_json import dataclass_json
from dotenv import load_dotenv

from . import config
from .utils.format_event_description import add_description_disclaimer

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


@dataclass_json
@dataclass
class EventData:
    date: datetime
    title: str
    is_event: bool
    description: str


def unix_time_to_human_date(unix_time: str) -> str:
    return datetime.fromtimestamp(int(float(unix_time))).strftime("%Y-%m-%d")


def get_calendar_prompt(date_ts: str, message: str) -> str:
    human_date = unix_time_to_human_date(date_ts)

    prompt = config.prompt.format(date=human_date, message=message)

    return prompt


# Define a function to send a message to ChatGPT 3.5
def execute_prompt(prompt: str) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message


def get_event_data(date_ts: str, message: str) -> EventData:
    if message == "":
        return EventData(date=None, title=None, is_event=False, description="")
    prompt = get_calendar_prompt(date_ts, message)
    response = execute_prompt(prompt)
    data = json.loads(response["content"])
    if data["is_event"]:
        # convert string to date and then back to UTC for later serialization
        date_naive = datetime.strptime(data["date"], "%Y-%m-%dT%H:%M:%S")
        date_local = pytz.timezone("US/Eastern").localize(date_naive)
        date_utc = date_local.astimezone(pytz.utc)
        data["date"] = date_utc
    else:
        data["date"] = None
    event_data = EventData(**data, description=message)

    return event_data


def get_event_data_summary(event_data: EventData, link: str) -> str:
    local_date = event_data.date.astimezone(pytz.timezone("US/Eastern"))
    human_readable_date = local_date.strftime("%A, %B %d, %Y %I:%M %p")
    disclaimer = add_description_disclaimer(link)
    summary = f"""*Title:* {event_data.title}\n*Date:* {human_readable_date} \n*Description:* {disclaimer}"""

    return summary
