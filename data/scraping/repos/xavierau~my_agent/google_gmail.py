# import os
import json
import os
from dataclasses import Field
from typing import List, Any

import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI

import dotenv

from utils.llm import get_response_content_from_gpt
from utils.logger import Logger
from googleapiclient.discovery import build
import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup

dotenv.load_dotenv()

from tools.common import Tool, ToolCallResult

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.readonly",
]


class GoogleGmailReader(Tool):
    """Search from Google"""
    name: str = "google_gmail_reader"
    description: str = "It is helpful when you need to search emails from gmail account."
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
                            "description": "The query you want to search. Take into account about the user preferences.",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Number of top result return. It must between 1 and 10. Without special reason, always set it to 3.",
                            "default": 3
                        }
                    },
                    "required": ["query", "limit"]
                }
            }
        }

    async def run(self, query: str, limit=5) -> ToolCallResult:

        Logger.info(f"tool:{self.name} query: {query}, limit: {limit}")

        print('search query: ', query)

        if query is None:
            raise Exception

        # url = f"https://customsearch.googleapis.com/customsearch/v1?cx={self.search_engine_id}&q={query}&key={self.key}&num={limit}"
        #
        # response_dict = requests.get(url).json()
        #
        # print("custom search response: ", response_dict)

        service = build(
            "customsearch", "v1", developerKey=os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        )

        res = (
            service.cse()
            .list(
                q=query,
                cx=os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
            )
            .execute()
        )

        items = res.get('items', [])

        if len(items) > limit:
            items = items[:limit]

        tasks = []

        for item in items:
            tasks.append(self._to_structure(query, item))

        summarized_websites = await asyncio.gather(*tuple(tasks))

        print("summarized_websites", summarized_websites)

        return ToolCallResult(result=json.dumps(summarized_websites))

    def _get_credentials(self) -> Any:
        """Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.
        """
        import os

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=8080)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds
