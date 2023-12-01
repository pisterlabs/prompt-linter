from __future__ import annotations

import logging
import os

import openai

from assistant.agent import Agent
from assistant.conversation import Conversation
from assistant.message import Message

LOG = logging.getLogger(__name__)
openai.api_key = os.environ["OPENAI_API_KEY"]


def add_comment_markers(text):
    # GPT still returns commentary even when I request it not to
    # This code can prefix those comments with '#', assuming the text is mostly well formatted

    robot_is_commenting = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("```"):
            lines[i] = "# " + line
            robot_is_commenting = not robot_is_commenting
        elif robot_is_commenting:
            lines[i] = "# " + line
        elif "```" in line:
            lines[i] = line.replace("```", "# ```")
    return "\n".join(lines)


class JupyterAssistantAgent(Agent):
    async def reply_to(self, conversation: Conversation) -> Message:
        message = conversation.last_message()
        prompt = (
            "The following is some Jupyter python code, its outputs, "
            "and a comment asking you to fill in some code. "
            "Please return the python code wrapped as ```python```:\n"
        )
        max_length = 10000 - len(prompt)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful Python Jupyter coding assistant.",
            },
            {
                "role": "user",
                "content": prompt + message.text[-max_length:],
            },
        ]
        LOG.info("Forwarding to ChatGPT:\n%s", messages)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
        )
        response_content = add_comment_markers(response.choices[0].message.content)
        LOG.info("ChatGPT replied: '%s'", response_content)
        return message.evolve(
            text=response_content,
            source="jupyter-assistant-plugin",
        )
