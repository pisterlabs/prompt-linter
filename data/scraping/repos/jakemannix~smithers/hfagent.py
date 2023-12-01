import datetime
import os
import pickle

from transformers.tools import OpenAiAgent
from transformers import Tool
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from transformers.utils import is_openai_available


if is_openai_available():
    import openai

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']


class GCalTool(Tool):
    name = "gcal_tool"
    description = "Google Calendar Tool: fetches upcoming events from the user's Google Calendar"

    inputs = ['calendar_id', 'time_min', 'time_max', 'max_results', 'order_by']
    outputs = ['text']
    repo_id = "gcal_tool"

    def __init__(self, api_key=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    '/tmp/credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        self.api_key = api_key
        self.service = build('calendar', 'v3', credentials=creds)

    def __call__(self, calendar_id='primary',
                 time_min=None, time_max=None, max_results=10, order_by='startTime'):
        if time_min is None:
            now = datetime.datetime.utcnow().isoformat() + 'Z'
        else:
            now = time_min
        events_result = self.service.events().list(calendarId=calendar_id, timeMin=now,
                                                   maxResults=max_results, singleEvents=True,
                                                   orderBy=order_by).execute()
        events = events_result.get('items', [])
        if not events:
            return 'No upcoming events found.'
        event_strings = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            event_strings.append(f"{start}: {event['summary']} ({event['description']})")
        return "\n".join(event_strings)


class MemoryStorageTool(Tool):
    name = "memory_storage_tool"
    description = "Memory Storage Tool: stores factual information in memory, along with a summary and a timestamp"

    inputs = ['summary', 'full_text', 'timestamp']
    outputs = []
    repo_id = "memory_storage_tool"

    def __init__(self, memory: list = None, api_key: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not is_openai_available():
            raise ImportError("Using `OpenAiAgent` requires `openai`: `pip install openai`.")

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "You need an openai key to use `OpenAIAgent`. You can get one here: Get one here "
                "https://openai.com/api/`. If you have one, set it in your env with `os.environ['OPENAI_API_KEY'] = "
                "xxx."
            )
        else:
            openai.api_key = api_key
        self.memory = memory or []

    def __call__(self, summary: str, full_text: str, timestamp: str):
        # FIXME: this call is not correct currently
        openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "text": summary},
                        {"role": "agent", "text": full_text}]
        )
        self.memory.append({
            'summary': summary,
            'full_text': full_text,
            'timestamp': timestamp
        })
        return None


class MemoryRetrievalTool(Tool):
    name = "memory_retrieval_tool"
    description = "Memory Retrieval Tool: searches through memory for relevant facts, " \
                  "optionally filtered by ranges of time"

    inputs = ['query', 'time_min', 'time_max']
    outputs = ['memory_struct']
    repo_id = "memory_retrieval_tool"

    def __init__(self, memory: list = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory or []

    def __call__(self, query: str, time_min: str = None, time_max: str = None):
        if time_min is None:
            time_min = datetime.datetime.min.isoformat()
        if time_max is None:
            time_max = datetime.datetime.max.isoformat()
        results = []
        for fact in self.memory:
            if time_min <= fact['timestamp'] <= time_max:
                if query in fact['full_text']:
                    results.append(fact)
        return results


def get_agent(api_key=None):
    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]
    tool = GCalTool()
    agent = OpenAiAgent(model="gpt-3.5-turbo-16k", api_key=api_key, additional_tools=[tool])
    return agent
