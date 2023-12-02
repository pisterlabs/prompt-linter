import datetime
import json
import logging
from typing import Any

import openai
import regex
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from main.decorators import session_authentication
from main.models import Events
from main.models import Tasks, TaskEnhancement
from main.models.events import EVENTS_FIELDS_MAPPING
from main.utils import model_to_dict, explore_nested_object

LOGGER = logging.getLogger("console")

CONTEXT_TO_ENHANCEMENT_TASK_MESSAGE = """
You are an expert Product Owner.
Your team uses SCRUM methodology.
The tasks are of technical nature.

Improve the task information:

- Suggest a better title if necessary.

- Suggest a better description or improve the existing one if necessary.
The description should be return in MARKDOWN with nice and detailed formatting.

- Write the task description with goal, business criteria (only when necessary), acceptance criteria and technical details (only when necessary)

- Provide a time span for the task in hours.

You ALWAYS return a JSON response with the following format:

{
      title: "string";
      description: string;
      duration: number;
};
"""

CONTEXT_TO_ENHANCEMENT_EVENT_MESSAGE = """
You are an expert Product Owner.
Your team uses SCRUM methodology.
Can you re organize the google calendar events basing your decision in tasks from JIRA cards?

Have in consideration the estimation, title and description of Jira CARDS and Google calendar events and avoid 
collisions with the existing events, the can not override each other.

You ALWAYS return a JSON response to create those events in google calendar api format including the issue_id in 
response under the same field name

Just return ONLY the list of object in JSON with key "events_to_create".
"""


@require_http_methods(["GET"])
@session_authentication()
def list_all(__: Any, task_id: int = None) -> JsonResponse:
    """
    List or details a task enhanced

    :param Any __: django request object
    :param int task_id: task id (issue_id)
    :return:
    """

    status = 200
    try:
        if task_id:
            response_data = model_to_dict(TaskEnhancement.objects.get(task__issue_id=task_id))
        else:
            response_data = {"result": [model_to_dict(item) for item in TaskEnhancement.objects.all()]}
    except TaskEnhancement.DoesNotExist:
        response_data = {"message": f"Task enhanced '{task_id}' does not exist"}
        status = 404
    except (AttributeError, Exception) as error:
        LOGGER.error(error)
        response_data = {"message": "Something went wrong during the process"}
        status = 400

    return JsonResponse(data=response_data, status=status)


@require_http_methods(["GET"])
@session_authentication()
def enhancement(request: Any, task_id: int) -> JsonResponse:
    """
    Enhance a given task.

    :param Any request: django request object
    :param int task_id: task id (issue_id)
    :return:
    """

    status = 200
    try:
        task = Tasks.objects.filter(issue_id=task_id).first()
        if not task:
            raise Tasks.DoesNotExist()

        openai.api_key = settings.OPENAPI_KEY
        messages = [
            {
                "role": "system",
                "content": CONTEXT_TO_ENHANCEMENT_TASK_MESSAGE
            },
            {
                "role": "user",
                "content": task.description or ""
            },
        ]

        chat = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )

        result = chat.choices[0].message.content
        task_enhanced, created = TaskEnhancement.objects.get_or_create(
            user=request.user,
            task=task,
        )

        task_enhanced.value = result
        task_enhanced.save()

        response_data = {"result": model_to_dict(task_enhanced)}
    except Tasks.DoesNotExist:
        response_data = {"message": f"Task '{task_id}' does not exist"}
        status = 404
    except (AttributeError, Exception) as error:
        LOGGER.error(error)
        response_data = {"message": "Something went wrong during the process"}
        status = 400

    return JsonResponse(data=response_data, status=status)


@require_http_methods(["GET"])
@session_authentication()
def enhancement_calendar_events(request: Any, task_id: int = None) -> JsonResponse:
    """
    Enhance a given task.

    :param Any request: django request object
    :param int task_id: task id (issue_id)
    :return:
    """

    status = 200
    try:
        now = datetime.datetime.now()
        end_date = (now + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        start_date = now.strftime("%Y-%m-%d")
        openai.api_key = settings.OPENAPI_KEY
        messages = [
            {
                "role": "system",
                "content": CONTEXT_TO_ENHANCEMENT_EVENT_MESSAGE
            },
            {
                "role": "user",
                "content": json.dumps({
                    "tasks_from_jira": [
                        model_to_dict(item, fields=["title", "description", "estimation", "priority", "issue_id"])
                        for item in Tasks.objects.all()
                    ],
                    "events_from_google_calendar": [
                        model_to_dict(item, fields=["start", "end"])
                        for item in Events.objects.filter(start__gte=start_date, end__lte=end_date)
                    ],
                })
            },
        ]

        chat = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )

        try:
            pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
            results = json.loads(pattern.findall(chat.choices[0].message.content)[0]).get("events_to_create") or []
            creds = Credentials.from_authorized_user_file(f"/tmp/{request.user.googlessouser.id}.json", [])
            service = build("calendar", "v3", credentials=creds)

            for calendar in service.calendarList().list().execute().get("items") or []:
                if calendar["accessRole"] == "owner":
                    calendar_id = calendar["id"]
                    break
            else:
                raise AttributeError("User didn't have any calendar with 'owner' access rights")

            new_calendar_events = []
            new_calendar_events_serialized = []
            for result in results:
                try:
                    new_event_res = service.events().insert(calendarId=calendar_id, body=result).execute()

                    new_event = {"user_id": request.user.id}
                    for model_field, field in EVENTS_FIELDS_MAPPING.items():
                        new_event[model_field] = explore_nested_object(new_event_res, field)

                    # there's no consistency in GPT answer so all these validations are "needed"
                    issue_id = (result.get("description") or "").lower().replace("issue_id:", "ID:").split("ID: ")[-1]
                    if not issue_id:
                        issue_id = result.get("id") or result.get("issue_id") or "not_found_such_thing"

                    associate_task = Tasks.objects.filter(issue_id=issue_id).first()
                    if associate_task:
                        new_event["task"] = associate_task

                    new_calendar_events.append(Events(**new_event))
                except Exception as error:
                    LOGGER.error(error)

            if new_calendar_events:
                new_calendar_events_serialized = [
                    model_to_dict(event)
                    for event in Events.objects.bulk_create(new_calendar_events)
                ]

            response_data = {"messages": "Events created with success", "results": new_calendar_events_serialized}
        except Exception as error:
            LOGGER.error(error)
            response_data = {
                "messages": "Unable to generate events on GCalendar",
                "results": chat.choices[0].message.content
            }
    except Tasks.DoesNotExist:
        response_data = {"message": f"Task '{task_id}' does not exist"}
        status = 404
    except (AttributeError, Exception) as error:
        LOGGER.error(error)
        response_data = {"message": "Something went wrong during the process"}
        status = 400

    return JsonResponse(data=response_data, status=status)
