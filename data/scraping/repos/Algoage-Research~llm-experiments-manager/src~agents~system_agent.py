import os
from typing import List
import openai

from logger import generate_logger
from utils.try_with_interval import safe_chat_complete

logger = generate_logger(__name__)


class SystemAgent():
    type = 'system'

    def __init__(self, system_config=dict):
        self.system_config = system_config
        if 'initial_message' in system_config:
            self.next_reply = system_config['initial_message']
        self.openai_conversation_history: List[dict] = []

    def output_text(self):
        return self.next_reply

    def input_text(self, text: str):
        reply = self._generate_output_text(text)
        self.next_reply = reply
        self.openai_conversation_history.append(
            {
                'role': 'user',
                'content': text
            }
        )
        self.openai_conversation_history.append(
            {
                'role': 'assistant',
                'content': reply
            }
        )
        return reply

    def is_finished(self):
        # fail safe for endless conversations
        return len(self.openai_conversation_history) >= 20

    def _generate_output_text(self, text: str):
        openai.api_key = os.environ['OPENAI_API_KEY']

        messages_for_openai: List[str] = [{
            'role': 'system',
            'content': self.system_config['system_prompt']
        }]

        messages_for_openai += self.openai_conversation_history + [{
            'role': 'user',
            'content': text
        }]

        result = safe_chat_complete(
            chatcompletion_kwargs=dict(
                model=self.system_config.get('openai_model', 'gpt-3.5-turbo'),
                messages=messages_for_openai
            )
        )
        reply = result.choices[-1].message
        logger.debug(reply)

        return reply.content
