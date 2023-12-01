# import os
import json
from dataclasses import Field
from datetime import datetime
from typing import List

import requests
import tiktoken
from bs4 import BeautifulSoup
from googleapiclient.errors import HttpError
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
import dotenv

from tools.google_calendar.main import search_events, create_event, update_event, delete_events
from utils.llm import get_response_content_from_gpt
from utils.logger import Logger

dotenv.load_dotenv()

from tools.common import Tool, ToolCallResult


class GetGoogleCalendarTool(Tool):
    """Search from Google"""
    name: str = "get_google_calendar_event"
    description: str = "It is helpful when you need read event from user's calendar"
    summary_model: str = "gpt-3.5-turbo"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query information for the event",
                        },
                        "timeMin": {
                            "type": "string",
                            "description": "Lower bound (exclusive) for an event's end time to filter by. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00Z.",
                            "default": datetime.now().isoformat() + "Z"
                        },
                        "timeMax": {
                            "type": "string",
                            "description": "Upper bound (exclusive) for an event's start time to filter by. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00Z",
                        },
                    },
                }
            }
        }

    async def run(self, query: str = None, timeMin: str = None, timeMax: str = None) -> ToolCallResult:
        Logger.info(f"tool:{self.name} query: {query}, timeMin: {timeMin}, timeMax: {timeMax}")

        events = search_events(query=query, timeMin=timeMin, timeMax=timeMax)

        return ToolCallResult(result=json.dumps(events))


class CreateGoogleCalendarTool(Tool):
    """Search from Google"""
    name: str = "create_google_calendar_event"
    description: str = "It is helpful when you need create a new event on user's calendar"
    summary_model: str = "gpt-3.5-turbo"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "The title of the event",
                        },
                        "location": {
                            "type": "string",
                            "description": "The location of the event. This can be a physical address or a virutal location such as a URL from Zoom or Teams.",
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the event",
                        },
                        "start": {
                            "type": "string",
                            "description": "When the event starts. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00Z.",
                        },
                        "end": {
                            "type": "string",
                            "description": "When the event ends. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00Z.",
                        },
                    },
                    "required": ["summary"]
                }
            }
        }

    async def run(self, summary: str, location: str = None, description: str = None, start: str = None,
            end: str = None) -> ToolCallResult:
        Logger.info(
            f"tool:{self.name} summary: {summary}, location: {location}, description: {description}, start: {start}, end: {end}")

        try:
            event = create_event(summary=summary, location=location, description=description, start=start, end=end)
            return ToolCallResult(result=json.dumps({
                "status": "success",
                "message": "Successfully created event",
                "event": event
            }))
        except HttpError as e:
            return ToolCallResult(result=json.dumps({
                "status": "error",
                "message": "Error creating event",
                "reason": e.reason
            }))


class ModifyGoogleCalendarTool(Tool):
    name: str = "update_google_calendar_event"
    description: str = "It is helpful when you need update existing event on user's calendar"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_id": {
                            "type": "string",
                            "description": "The id of the event to update",
                        },
                        "summary": {
                            "type": "string",
                            "description": "The title of the event",
                        },
                        "location": {
                            "type": "string",
                            "description": "The location of the event. This can be a physical address or a virutal location such as a URL from Zoom or Teams.",
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the event",
                        },
                        "start": {
                            "type": "string",
                            "description": "When the event starts. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00Z.",
                        },
                        "end": {
                            "type": "string",
                            "description": "When the event ends. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00Z.",
                        },
                    },
                    "required": ["event_id"]
                }
            }
        }

    async def run(self, event_id: str, summary: str = None, location: str = None, description: str = None, start: str = None,
            end: str = None) -> ToolCallResult:
        Logger.info(
            f"tool:{self.name} event_id: {event_id}, summary: {summary}, location: {location}, description: {description}, start: {start}, end: {end}")

        try:
            event = update_event(event_id=event_id,
                                 summary=summary,
                                 location=location,
                                 description=description,
                                 start=start,
                                 end=end)
            return ToolCallResult(result=json.dumps({
                "status": "success",
                "message": "Successfully updated event",
                "event": event
            }))
        except HttpError as e:
            return ToolCallResult(result=json.dumps({
                "status": "error",
                "message": "Error updating event",
                "reason": e.reason
            }))


class DeleteGoogleCalendarTool(Tool):
    name: str = "delete_google_calendar_event"
    description: str = "It is helpful when you need to delete existing event on user's calendar"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_ids": {
                            "type": "array",
                            "description": "The id of the event to update",
                            "items": {
                                "type": "string",
                                "description": "The id of the event to delete",
                            }
                        },
                        "confirmed": {
                            "type": "boolean",
                            "description": "You have ask the user and user has explicilty confirm the deletion of the events",
                            "default": False,
                        }
                    },
                    "required": ["event_ids", "confirmed"]
                }
            }
        }

    async def run(self, event_ids: List[str], confirmed: bool = False) -> ToolCallResult:

        if not confirmed:
            return ToolCallResult(result=json.dumps({
                "status": "warning",
                "message": "Ask user to confirm the event deletion",
            }))

        try:
            delete_events(event_ids=event_ids)
            return ToolCallResult(result=json.dumps({
                "status": "success",
                "message": "Successfully delete event(s)",
            }))
        except HttpError as e:
            return ToolCallResult(result=json.dumps({
                "status": "error",
                "message": "Error deleting event",
                "reason": e.reason
            }))
