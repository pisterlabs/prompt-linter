import datetime
import os
import re
import json

import openai

from typing import List, Literal, TypedDict
from agents.system_agent import SystemAgent

from logger import generate_logger
from utils.text import extract_json_from_string
from utils.try_with_interval import safe_chat_complete

logger = generate_logger(__name__)


class UserEvent(TypedDict):
    time: datetime
    received_from: str
    sender_type: str


class UserAgent():
    type = 'user'

    def __init__(self, user_config):
        self.user_config = user_config
        self.conversation_log: List[UserEvent] = []
        self.success = False
        self.finished_conversation = False

    def input_text(self, text: str, sender: SystemAgent):
        self.conversation_log.append({
            'time': datetime.datetime.now(),
            'text': text,
            'sender_type': sender.type,
        })

    def output_text(self):
        text = self._generate_reply()
        self.conversation_log.append({
            'time': datetime.datetime.now(),
            'text': text,
            'sender_type': self.type,
        })
        return text

    def is_finished(self):
        return self.finished_conversation or len(self.conversation_log) >= 20

    def _generate_reply(self):
        openai.api_key = os.environ['OPENAI_API_KEY']

        instruction_prompt = f'''
# Instruction
{self.user_config.get('instruction', '')}

## Your persona
{self.user_config.get('persona', '')}

'''

        messages_for_openai: List[str] = []
        messages_for_openai.append({
            'role': 'system',
            'content': instruction_prompt
        })

        messages_for_openai += [
            {
                'role': 'assistant' if conversation_log['sender_type'] == 'user' else 'user', # Flip the roles since we are generating user's input.
                'content': conversation_log['text']
            } for conversation_log in self.conversation_log
        ]
        logger.debug(self.conversation_log)

        output_format_json_dict = {
            'success': f'boolean: {self.user_config["definition_of_success"]}',
            'reply': 'string: reply to service. empty means user won\'t reply to service because you finished the conversation or you want to ignore its message because not satisfied',
        }

        messages_for_openai[-1]['content'] = f'''
{instruction_prompt}

# Output Format (explicitly follow the json format below)

{json.dumps(output_format_json_dict, indent=2)}

# User Input
{messages_for_openai[-1]['content']}
        '''

        result = safe_chat_complete(
            chatcompletion_kwargs=dict(
                model=self.user_config.get('openai_model', 'gpt-3.5-turbo'),
                messages=messages_for_openai,
                temperature=0
            )
        )
        logger.debug(messages_for_openai)
        reply = result.choices[-1].message.content
        logger.debug(reply)

        reply_dict = extract_json_from_string(reply)

        if reply_dict is not None:
            self.success = reply_dict.get('success', self.success)
            reply_str = reply_dict.get('reply', reply)
        else:
            reply_str = re.sub(r'\{.*\}', '', reply, re.DOTALL).split(':')[-1]

        # if reply is empty, it stops the conversation
        if self.finished_conversation or not bool(reply_str.strip()):
            self.finished_conversation = True
            return None
        else:
            return reply_str
