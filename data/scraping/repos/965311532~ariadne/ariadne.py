import json
import time
from typing import Optional

from openai import OpenAI
from openai.types.beta import Assistant, Thread

from config import ARIADNE_OPENAI_ASSISTANT_ID, ARIADNE_OPENAI_THREAD_ID

ARIADNE_SYSTEM_PROMPT = """### ROLE

You are Ariadne, an advanced digital business advisor, created to help the TESEO team grow their startup and increase their profitability. Your role is to give them expert advice, clear strategies, and recommendations based on hard data that help them achieve their business goals.

### OPERATIONAL GUIDELINES

- Give TESEO clear steps backed by quantitative analysis that directly contribute to TESEO's strategic objectives.
- Always look for what information might be missing and ask specific questions to get it.
- Ignore messages that don't require an answer by replying with "NO_RESPONSE" and focus on giving useful answers to questions that do matter.
- Offer your advice with confidence. Work with the TESEO team closely and don't be afraid to question what they think they know.
- Make sure your advice stands on solid numbers and facts, and ask for any key details you don't have.
- Keep your language straightforward and to the point, so you're easy to understand. Don't use irrelevant pleasentries or jargon.
- Reference and quote previous discussions to contextualize your response.
- You are part of the TESEO team. Use words like "we" and "us" to show that you're working together with them.

### RESPONSE FORMATTING

- Your answers will be forwared to the TESEO team as emails, so use a professional tone and format them accordingly.
- Compose emails using Markdown for straightforward text formatting. (Specifically, I'll be using the `markdown` Python library to convert your answers to HTML, so make sure you use the correct syntax for that).
- Use bold, italics, structured paragraphs, and clear headings judiciously to enhance readability without compromising the professional tone of your emails.
- Do not put "---" at the end or beginning of your emails.
- Do not use nested list (they don't render well in emails).
- Insert links with Markdown using the following format: [link text](link URL).
- Make sure your email replies are just the message with no extra bits like subject lines or other email parts, and double-check that they sound right and are easy to read.
- Keep your answers short and clear, adjusting detail to the complexity of the issue.
- End with a specific call to action or question(s) to keep the conversation moving.
"""

ARIADNE_PROMPT_TEMPLATE = """Latest Email:
From: {from_}
To: {to}
CC: {cc}
Subject: "{subject}"
Date: {date}

---
{body}
---

Upon review of the above email, please offer your insights and recommendations. If you deem that a response is not required, please indicate with "NO_RESPONSE"."""


class AriadnePrompt:
    def __init__(self, email: dict):
        self.email = email

    def build(self) -> str:
        """Fills the Ariadne prompt with the email's data."""
        cc = self.email.get("cc", "none")  # If there is no cc, set it to "none"
        return ARIADNE_PROMPT_TEMPLATE.format(**{**self.email, "cc": cc})


class Ariadne:
    def __init__(self, openai_client: OpenAI, debug=False):
        self.openai_client = openai_client
        self.debug = debug
        # If the thread a IDs is not provided, create it
        self.openai_thread = self._get_thread(ARIADNE_OPENAI_THREAD_ID)
        self.openai_assistant = self._get_synced_assistant(ARIADNE_OPENAI_ASSISTANT_ID)
        # Print the IDs in the logs for debugging
        print(f"DEBUG: debug mode: {self.debug}")
        print(f"DEBUG: OpenAI Thread ID: {self.openai_thread.id}")
        print(f"DEBUG: OpenAI Assistant ID: {self.openai_assistant.id}")

    def _get_synced_assistant(self, assistant_id: Optional[str] = None) -> Assistant:
        """Get the OpenAI Assistant instance. If the ID is not provided, it will create it."""
        # If the assistant ID is not provided, create it
        if not assistant_id:
            return self.openai_client.beta.assistants.create(
                name="Ariadne AI",
                instructions=ARIADNE_SYSTEM_PROMPT,
                tools=[{"type": "retrieval"}],
                model="gpt-4-1106-preview",
            )

        # If it is provided, get it and make sure the system prompt is up to date
        assistant_instance = self.openai_client.beta.assistants.retrieve(
            assistant_id=assistant_id
        )
        if assistant_instance.instructions != ARIADNE_SYSTEM_PROMPT:
            # Update the system prompt
            self.openai_client.beta.assistants.update(
                assistant_id=assistant_id, instructions=ARIADNE_SYSTEM_PROMPT
            )
        return assistant_instance

    def _get_thread(self, thread_id: Optional[str] = None) -> Thread:
        """Get the OpenAI Thread instance. If the ID is not provided, it will create it.
        If Ariadne is set up in debug mode, this will always create a new thread."""
        # If the thread ID is not provided, create it
        if not thread_id or self.debug:
            return self.openai_client.beta.threads.create(
                messages=self._get_initial_messages()
            )
        # If it is provided, get it
        return self.openai_client.beta.threads.retrieve(thread_id=thread_id)

    def _get_initial_messages(self) -> list:
        """Get the initial messages for the thread."""
        # Parse the `emails.json` file (already in the format expected by the OpenAI API)
        with open("emails.json", "r") as f:
            return json.load(f)

    def get_reply(self, message: str) -> str:
        """Sends a message to the Assistant and returns the Assistant's response."""
        # 1. Send a new message to the assistant thread
        self.openai_client.beta.threads.messages.create(
            thread_id=self.openai_thread.id, role="user", content=message
        )

        # 2. Start the assistant thread run
        run = self.openai_client.beta.threads.runs.create(
            thread_id=self.openai_thread.id,
            assistant_id=self.openai_assistant.id,
        )

        # 3. Wait for the assistant thread run to complete
        while run.status != "completed":
            run = self.openai_client.beta.threads.runs.retrieve(
                thread_id=self.openai_thread.id, run_id=run.id
            )
            time.sleep(1)  # Wait 1 second before checking again

        # 4. Get the assistant's response from the messages
        messages = self.openai_client.beta.threads.messages.list(
            thread_id=self.openai_thread.id, limit=1  # Only get the last message
        )

        # 5. Extract the assistant's response from the messages
        answer = messages.data[0].content[0].text.value

        return answer
