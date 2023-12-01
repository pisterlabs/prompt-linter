import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import openai
import json


class GCalActionLayer:
    def __init__(self):
        self.scopes = ["https://www.googleapis.com/auth/calendar"]
        self.creds = None
        if os.path.exists("token.json"):
            self.creds = Credentials.from_authorized_user_file(
                "token.json", self.scopes
            )

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", self.scopes
                )
                self.creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(self.creds.to_json())
        self.service = build("calendar", "v3", credentials=self.creds)


class OpenAIActionLayer:
    def __init__(self):
        with open("openai_conf.json", "r") as f:
            conf = json.load(f)
            self.api_key = conf["key"]
            self.model = conf["model"]
            self.role = conf["role"]
        self.service = self._create_service()

    def _create_service(self):
        openai.api_key = self.api_key
        # create service function that takes in a prompt and returns a response
        def service(prompt: str):
            messages = [{"role": self.role, "content": prompt}]
            response = openai.ChatCompletion.create(
                model=self.model, messages=messages, temperature=0,
            )
            print(response.choices[0].message["content"])
            return response.choices[0].message["content"]

        return service


class EclairGCalAction:
    def __init__(self, ctx):
        self.gcal = GCalActionLayer()
        self.openai = OpenAIActionLayer()
        self.ctx: dict = ctx  # not sure what the dtype will be, i'll assume the text is already extracted
        self.prompt = (
            """
**Prompt:**
Extract meeting information from Slack messages. Current date: {1}. Identify the final decided info between the two participants. If information is missing, simply set it to 'None'. Only respond in the provided format, and count virtual meetings as "location":"virtual". Additionally, the default duration of a meeting is 1 hour.

**Message:**
{2}

**Response format:**
{
'location': str
'start_time': YYYY-MM-DDTHH:MM
'duration': number of hours
'topic': str
}
""".replace(
                "{1}",
                datetime.datetime.now().strftime("%Y-%m-%d")
                + f"Day of the week: {datetime.datetime.now().weekday()}",
            )
            .replace("{2}", self.ctx["in_text"])
            .replace("'", '"')
        )
        self.meeting_info: dict = json.loads(self.openai.service(self.prompt))
        self.meeting_info["attendees"] = self.ctx["attendees"]
        self.meeting_info["start_time"] = datetime.datetime.strptime(
            self.meeting_info["start_time"], "%Y-%m-%dT%H:%M"
        )
        self.meeting_info["duration"] = float(self.meeting_info["duration"])

    def format_event(self):
        print("Formatting event")
        event = {
            "summary": self.meeting_info["topic"],
            "location": self.meeting_info["location"],
            "start": {
                "dateTime": self.meeting_info["start_time"].isoformat(),
                "timeZone": "America/Los_Angeles",
            },
            "end": {
                "dateTime": (
                    self.meeting_info["start_time"]
                    + datetime.timedelta(hours=self.meeting_info["duration"])
                ).isoformat(),
                "timeZone": "America/Los_Angeles",
            },
            "attendees": [
                {"email": attendee} for attendee in self.meeting_info["attendees"]
            ],
        }
        print("Event formatted")
        print(event)
        return event

    def create_event(self):
        event = self.format_event()
        print(event)
        event = (
            self.gcal.service.events()
            .insert(calendarId="primary", body=event)
            .execute()
        )
        print("Event created: %s" % (event.get("htmlLink")))

    def __repr__(self) -> str:
        return f"EclairGCalAction({self.meeting_info}))"

# Example usage:
# def main():
#     ctx = {
#         "attendees": ["beckett.leemesser@gmail.com"],
#         "in_text": "drop by Room 325 in the Gates computer science building from 12-4:30 next wednesday so we can discuss the project",
#     }
#     action = EclairGCalAction(ctx)
#     action.create_event()

# if __name__ == "__main__":
#     main()
