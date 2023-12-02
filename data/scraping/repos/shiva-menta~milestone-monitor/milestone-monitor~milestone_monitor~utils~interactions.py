import os
import pytz
from typing import Tuple
from django.conf import settings
from django.db.models import F, Max, OuterRef, Q, Subquery, Value
from django.utils.dateparse import parse_datetime, parse_time

import cohere
import pinecone

from milestone_monitor.models import User, Goal, Importance, Frequency
from .constants import (
    str_to_frequency,
    str_to_importance,
    frequency_to_str,
    importance_to_str,
)
from datetime import datetime, timedelta
from django.forms.models import model_to_dict

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
COHERE_EMBED_MODEL = "embed-english-light-v2.0"


def get_and_create_user(phone_number: str):
    user = None
    try:
        user = User.objects.get(phone_number=phone_number)
    except:
        user = User(phone_number=phone_number)
        user.save()

    return user


def create_goal(goal_data: dict, phone_number: str):
    print("Goal data:")
    print(goal_data)

    """
    Creates a goal based on input data for a specific user.

    NEED TIMEZONE FEATURE

    goal_data: – can we move this somewhere else potentially – looks kind of messy right here
      - name: string
      - description: string
      - estimatedImportance: 'HIGH' | 'MEDIUM' | 'LOW'
      - estimatedDurationHours: int
      - goalFrequency: 'DAILY' | 'WEEKLY' | None
      - reminderFrequency: 'HOURLY' | 'DAILY' | 'WEEKLY' | 'BIWEEKLY' | 'MONTHLY' | None
      - reminderTime: HH:MM
      - status: 'SUCCESS'
      - dueDate: datetime
      - isRecurring: 0 | 1
    user: string (of the form "+12345678901")
    """
    # Step 1: get current user
    parsed_user_number = int(phone_number[1:])
    user = get_and_create_user(parsed_user_number)

    # Step 2: format common necessary data for goal
    # Assume that the naive datetime is in a certain timezone (e.g., New York)
    fields = {
        "user": user,
        "title": goal_data["name"],
        "description": goal_data["description"],
        "importance": str_to_importance.get(
            goal_data["estimatedImportance"], Importance.LOW
        ),
        "end_at": goal_data["dueDate"],
        "completed": False,
    }

    if "dueDate" in goal_data and goal_data["dueDate"]:
        end_at = parse_datetime(goal_data["dueDate"])
        ny_tz = pytz.timezone("America/New_York")
        aware_dt = ny_tz.localize(end_at)
        utc_dt = aware_dt.astimezone(pytz.UTC)
        fields["end_at"] = utc_dt
    if goal_data["reminderFrequency"] and goal_data["reminderTime"]:
        fields["reminder_start_time"] = datetime.strptime(
            goal_data["reminderTime"], "%H:%M"
        )
        fields["reminder_frequency"] = str_to_frequency.get(
            goal_data["reminderFrequency"], Frequency.DAILY
        )

    # Step 3: actually create and save goal
    g = Goal(**fields)
    g.save()

    goal_id = g.id

    # Step 4: add goal embeddings + info to pinecone
    create_goal_pinecone(
        goal_id=goal_id,
        goal_description=goal_data["name"] + ": " + goal_data["description"],
        user=str(user.phone_number),
    )

    return goal_id


def modify_goal(goal_id: int, data=dict):
    """
    Modifies current goal based on a dict of information. Models' modify functions
    check against its attribute keys. Functions can handle almost any combination of
    inputs, allowing for flexible modify queries.
    """
    goal_instance = Goal.objects.get(id=goal_id)
    goal_instance.modify(data)


def get_goal_info(goal_id: int):
    goal_instance = Goal.objects.get(id=goal_id)

    # Construct goal info string
    goal_info = model_to_dict(goal_instance)
    goal_info.pop("id", None)
    goal_info.pop("name", None)
    goal_info.pop("user", None)
    goal_info.pop("reminder_task", None)
    goal_info.pop("final_task_id", None)

    print(goal_info)

    if (
        "reminder_start_time" in goal_info
        and goal_info["reminder_start_time"] is not None
    ):
        goal_info["reminder_start_time"] = datetime.strftime(
            goal_info["reminder_start_time"], "%m/%d/%Y, %H:%M:%S"
        )
    if "end_at" in goal_info and goal_info["end_at"] is not None:
        goal_info["end_at"] = datetime.strftime(
            goal_info["end_at"], "%m/%d/%Y, %H:%M:%S"
        )
    if "importance" in goal_info and goal_info["importance"] is not None:
        goal_info["importance"] = importance_to_str[goal_info["importance"]]
    if (
        "reminder_frequency" in goal_info
        and goal_info["reminder_frequency"] is not None
    ):
        goal_info["reminder_frequency"] = frequency_to_str[
            goal_info["reminder_frequency"]
        ]

    return "Goal info: " + str(goal_info)


def get_all_goals() -> str:
    response = "List of all goals (may be empty if there are no goals):"

    all_goals = Goal.objects.all()
    for goal in all_goals:
        response += f"- Goal name: {goal.title}, goal id: {goal.id}\n"

    return response


def create_goal_pinecone(goal_id: int, goal_description: str, user: str):
    print(goal_id, goal_description, user)

    """
    Adds the goal to pinecone

    goal_id: django id for the goal as an int
    goal_description: string description of the goal
    user: user's phone number as a string (no plus)
    """

    # Retrieve embedding from description via cohere
    co = cohere.Client(COHERE_API_KEY)
    embeds = co.embed(
        texts=[goal_description], model=COHERE_EMBED_MODEL, truncate="LEFT"
    ).embeddings

    # Open up pinecone connection (temp)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(PINECONE_INDEX)

    # Set metadata
    # We're entering the title and desc as the input
    # as opposed to an "update"
    metadata = {
        "type": "title_and_desc",
    }

    # Pinecone expects a string ID
    vector_item = (str(goal_id), embeds[0], metadata)

    # Add goal to pinecone
    index.upsert(vectors=[vector_item], namespace=user)

    print(">>> Successfully added goal to Pinecone")

    return True


def retrieve_goal_pinecone(query: str, user: str) -> int:
    """
    Retrieves a goal ID from Pinecone based on semantic similarity, filtered by user

    Output: goal ID
    """

    co = cohere.Client(COHERE_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(PINECONE_INDEX)

    xq = co.embed(texts=[query], model=COHERE_EMBED_MODEL, truncate="LEFT").embeddings

    res = index.query(
        xq,
        top_k=1,
        include_metadata=False,
        filter={"type": {"$eq": "title_and_desc"}},
        namespace=user,
    )
    print(res)
    return int(res["matches"][0]["id"])


# def update_goal_notes_pinecone(
#     added_notes: str, goal_id: int, goal_type: int, user: str
# ):
#     co = cohere.Client(COHERE_API_KEY)
#     pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
#     index = pinecone.Index(PINECONE_INDEX)

#     xq = co.embed(texts=[added_notes], model=COHERE_EMBED_MODEL, truncate="LEFT").embeddings
