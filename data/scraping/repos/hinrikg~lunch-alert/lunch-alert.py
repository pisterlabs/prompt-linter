from datetime import datetime, time
import json
import logging
import os
import sys

from dateutil.tz import UTC
from icalevents import icalevents
import requests


LUNCH_CALENDAR_URL = os.environ["LUNCH_CALENDAR_URL"]
LUNCH_CALENDAR_BASIC_AUTH_TOKEN = os.environ.get(
    "LUNCH_CALENDAR_BASIC_AUTH_TOKEN", None
)
HOLIDAY_CALENDAR_URL = os.environ["HOLIDAY_CALENDAR_URL"]
HOLIDAY_CALENDAR_BASIC_AUTH_TOKEN = os.environ.get(
    "HOLIDAY_CALENDAR_BASIC_AUTH_TOKEN", None
)
SLACK_URL = os.environ["SLACK_URL"]
DATETIME_OVERRIDE = os.environ.get("DATETIME_OVERRIDE", None)
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY", None)

MENU_MESSAGE = "Good morning everyone. For lunch today we're having {}"
LUNCH_MESSAGE = "<!here> It's lunchtime!"


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(argv):
    logger.info("starting with args {}".format(argv))

    if DATETIME_OVERRIDE:
        logger.info("pretending that today is {}".format(today()))

    if is_the_weekend():
        logger.info("stopping - it's the weekend")
        return

    command = argv[0]

    if command == "menu":
        logger.info("announcing today's menu")
        menu()
    elif command == "lunch":
        logger.info("announcing lunch time")
        lunch()
    else:
        logger.info("unknown command {}".format(command))


def menu():
    holiday_event = fetch_holiday_event()
    if holiday_event:
        logger.info("stopping - it's a holiday")
        return

    lunch_events = fetch_lunch_events()
    if lunch_events is None or len(lunch_events) == 0:
        logger.info("stopping - there's nothing on the menu")
        return

    if len(lunch_events) > 1:
        logger.info("stopping - menu has two or more events")
        return

    send_menu_message(lunch_events[0])


def lunch():
    holiday_event = fetch_holiday_event()
    if holiday_event:
        logger.info("stopping - happy holidays!")
        return

    send_message(LUNCH_MESSAGE)


def is_the_weekend():
    return now().isoweekday() >= 6


def fetch_lunch_events():
    events = fetch_events_today(
        LUNCH_CALENDAR_URL, basic_auth_token=LUNCH_CALENDAR_BASIC_AUTH_TOKEN
    )

    # The menu calendar is notoriously unreliable in terms of event data accuracy.
    # Sometimes the events are all day events, sometimes they have a start time,
    # sometimes the start time is incorrect. Sometimes the event summary is numbered
    # so that breakfast is "1." and lunch is "2.", sometimes this numbering is reversed
    # or repeated.

    # We deal with this inaccuracy by scoring the events according to couple of very
    # simple rules and then we select the event with the highest score. Here is an
    # example of the scoring:

    # 12:00    2. summary   4
    # 12:00    summary      3
    # 12:00    1. summary   2
    # all day  2. summary   1
    # all day  summary      0
    # all day  1. summary  -1
    # 8:00     2. summary  -2
    # 8:00     summary     -3
    # 8:00     1. summary  -4

    scored_events = []
    for event in events:
        score = 0

        if starts_around_lunch(event):
            score += 3
        elif not event.all_day:
            score -= 3

        if event.summary.startswith("2."):
            score += 1
        elif event.summary.startswith("1."):
            score -= 1

        logger.info("event {} got score {}".format(event, score))
        scored_events.append((score, event))

    sorted_events = sorted(scored_events, reverse=True)
    if len(sorted_events) > 1 and sorted_events[0][0] == sorted_events[1][0]:
        return [event for _, event in sorted_events[:2]]
    elif sorted_events:
        return [sorted_events[0][1]]


def starts_around_lunch(event):
    return not event.all_day and today_at(11) < event.start < today_at(13)


def fetch_holiday_event():
    events = fetch_events_today(HOLIDAY_CALENDAR_URL)
    # let's just naively use the first available entry
    return events[0] if events else None


def fetch_events_today(url, basic_auth_token=None):
    # for some reason icalevents thinks it's cute to return all-day events from the
    # day before (and sometimes after) the requested start date, so we need to filter
    # those out manually.
    logger.info("fetch_events_today %s", url)
    events = [
        event
        for event in _fetch_events_with_retry(
            url, start=today(), basic_auth_token=basic_auth_token
        )
        if event.start.date() == today()
    ]
    logger.info("found %s events:", len(events))
    for i, event in enumerate(events):
        logger.info("%s: %s", i, event)
    return events


def _fetch_events_with_retry(url, start, basic_auth_token=None, retries=3):
    attempts = 0
    fetched_events = None
    while fetched_events is None and attempts <= retries:
        try:
            headers = {}
            if basic_auth_token:
                headers["Authorization"] = "Basic {}".format(basic_auth_token)
            response = requests.get(url, headers=headers)
            fetched_events = icalevents.events(
                string_content=response.content, start=start
            )
        except TimeoutError:
            logger.warning("request timed out")
            attempts += 1
            if attempts > retries:
                logger.error("giving up after %s retries", retries)
                raise
    return fetched_events


def send_menu_message(event):
    logger.info("send_menu_message %s", event)
    send_message(get_lunch_summary(event))


def get_lunch_summary(event):
    if OPEN_AI_API_KEY:
        logger.info("We have an OpenAI API key! Let's delegate some work to the AI ...")
        try:
            return get_ai_menu_announcement(
                api_key=OPEN_AI_API_KEY, menu_text=event.summary
            )
        except Exception:
            logger.exception("The AI failed")
    return MENU_MESSAGE.format(event.summary)


def send_message(text):
    logger.info("send_message %s", text)
    requests.post(SLACK_URL, json={"text": text})


def now():
    if DATETIME_OVERRIDE:
        return datetime.strptime(DATETIME_OVERRIDE, "%Y-%m-%d")
    return datetime.utcnow()


def today():
    return now().date()


def today_at(hour):
    return datetime.combine(today(), time(hour=hour, tzinfo=UTC))


def get_ai_menu_announcement(api_key, menu_text):
    import openai

    openai.api_key = api_key

    messages = json.load(open("prompts/menu_prompt_context.json"))
    messages.append(
        {
            "role": "user",
            "content": menu_text,
        }
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception:
        logger.exception("Uncaught exception")
