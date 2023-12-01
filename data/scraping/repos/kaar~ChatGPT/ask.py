#!/usr/bin/env python3

import os

from open_ai_chat import OpenAiChatClient

OPENAI_SESSION_TOKEN = os.environ["OPENAI_SESSION_TOKEN"]

if not OPENAI_SESSION_TOKEN:
    raise ValueError("Missing OPENAI_SESSION_TOKEN")

client = OpenAiChatClient(OPENAI_SESSION_TOKEN)

print(client.conversation(input()).text)
