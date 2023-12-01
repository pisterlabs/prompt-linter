from typing import Optional

import json
from itertools import groupby
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage
from llm_task_handler.handler import OpenAIFunctionTaskHandler
from llm_task_handler.handler import ProgressMessageFunc
from llm_task_handler.handler import TaskState

from secretary.database import remove_account


class DisconnectAccount(OpenAIFunctionTaskHandler):
    def task_type(self) -> str:
        return 'handle_account_disconnection_or_termination_request'

    def intent_selection_function(self) -> dict:
        return {
            'name': self.task_type(),
            'description': 'Help user disconnect/terminate/cancel their account',
            'parameters': {
                'type': 'object',
                'properties': {}
            }
        }

    async def transition(
        self,
        cur_state: TaskState,
        progress_message_func: Optional[ProgressMessageFunc] = None,
    ) -> TaskState:
        function = {
            'name': 'identify_account_disconnection_or_termination_messages',
            'description': 'Help user disconnect their account',
            'parameters': {
                'type': 'object',
                'properties': {
                    'encoded_conversation': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                            'enum': ['A', 'B', 'C', 'D', 'E'],
                        },
                        'description': '''
Account disconnection requires analysis of these message types:

A. User requests account disconnection
B. AI asks for confirmation
C. Same user confirms yes
D. Same user confirms no
E. Other

Detect these types and retain original message order in the resulting array.
'''
                    },
                },
            }
        }

        chat_model = ChatOpenAI(  # type: ignore
            model_name='gpt-4',
            temperature=0,
            max_tokens=250,
            model_kwargs={"functions": [function]},
        )

        model_reply = chat_model([
            HumanMessage(content=cur_state.user_prompt),
        ])

        func_args = json.loads(model_reply.additional_kwargs['function_call']['arguments'])

        encoded_convo = func_args['encoded_conversation']
        cleaned_convo = [t for t in encoded_convo if t != 'E']
        cleaned_convo = [t for t, _ in groupby(cleaned_convo)]

        if encoded_convo[-1:] == ['A']:
            reply = '''
Can you confirm that you want to disconnect your account now?
'''
        elif cleaned_convo[-3:] == ['A', 'B', 'C']:
            remove_account(self.user_id)

            reply = 'Your account has been disconnected'
        elif cleaned_convo[-3:] == ['A', 'B', 'D']:
            reply = 'Okay, I won\'t disconnect your account.'
        else:
            reply = 'What?'

        return TaskState(
            handler=self,
            user_prompt=cur_state.user_prompt,
            reply=reply,
            is_done=True,
        )


class DisconnectAccountAbort(OpenAIFunctionTaskHandler):
    def task_type(self) -> str:
        return 'abort_account_disconnection_request'

    def intent_selection_function(self) -> dict:
        return {
            'name': self.task_type(),
            'description': 'Abort the account disconnection',
            'parameters': {
                'type': 'object',
                'properties': {}
            }
        }

    async def transition(
        self,
        cur_state: TaskState,
        progress_message_func: Optional[ProgressMessageFunc] = None,
    ) -> TaskState:
        return TaskState(
            handler=self,
            user_prompt=cur_state.user_prompt,
            reply='Okay, I won\'t disconnect your account.',
            is_done=True,
        )
