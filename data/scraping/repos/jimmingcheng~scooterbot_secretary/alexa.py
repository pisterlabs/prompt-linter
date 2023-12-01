from typing import NamedTuple

import arrow
import json
import logging
import openai
from ask_sdk_core.skill_builder import CustomSkillBuilder
from ask_sdk_core.api_client import DefaultApiClient
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_model.intent import Intent
from ask_sdk_model.slot import Slot
from ask_sdk_model.dialog.delegate_directive import DelegateDirective
from ask_sdk_runtime.exceptions import DispatchException
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.utils import is_intent_name
from ask_sdk_core.utils import is_request_type

from secretary.calendar import event_start_time
from secretary.write import add_todo


class ChatGPTOutput(NamedTuple):
    description: str
    date: arrow.Arrow
    confirmation_message: str


class AddTodoHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name('AddTodo')(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        access_token = str(handler_input.request_envelope.context.system.user.access_token)  # type: ignore
        intent = handler_input.request_envelope.request.intent  # type: ignore
        user_prompt = intent.slots['Prompt'].value  # type: ignore

        chatgpt_output = self.get_chatgpt_output(user_prompt)

        todo, reminder_days_before = add_todo(access_token, chatgpt_output.description, chatgpt_output.date)
        todo_start_time = event_start_time(todo)

        sentences = [chatgpt_output.confirmation_message]

        if reminder_days_before > 0:
            duration = todo_start_time.humanize(
                other=todo_start_time.shift(days=-reminder_days_before),
                only_distance=True
            )
            sentences.append(f"I'll remind you {duration} before.")

        speech = ' '.join(sentences)

        handler_input.response_builder.speak(speech).set_should_end_session(True)
        return handler_input.response_builder.response

    def get_chatgpt_output(self, user_prompt: str) -> ChatGPTOutput:
        instructions_prompt = f"""
Current time is {arrow.now()}

The user will input a todo list item in plain English. Convert it to this format:
{{"description": "Cut my hair", "date": "YYYY-MM-DD", "confirmation_message": "<Succinct spoken confirmation that the described todo is recorded on the supplied date>"}}

Output only the json.
        """

        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': instructions_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )

        completion_text = completion.choices[0].message.content
        print(completion_text)

        completion_data = json.loads(completion_text)

        return ChatGPTOutput(
            description=completion_data['description'],
            date=arrow.get(completion_data['date']),
            confirmation_message=completion_data['confirmation_message'],
        )


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type('LaunchRequest')(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        updated_intent = Intent(
            name='AddTodo',
            slots={
                'Prompt': Slot(name='Prompt', value=None),
            }
        )
        handler_input.response_builder.add_directive(DelegateDirective(updated_intent))
        return handler_input.response_builder.response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input: HandlerInput, exception: Exception) -> bool:
        return False

    def handle(self, handler_input: HandlerInput, exception: Exception) -> Response:
        if not isinstance(exception, DispatchException):
            logging.exception('Exception while responding to Alexa request')

        handler_input.response_builder.speak('Sorry, something went wrong.').set_should_end_session(True)
        return handler_input.response_builder.response


def get_skill_builder() -> SkillBuilder:
    sb = CustomSkillBuilder(api_client=DefaultApiClient())
    sb.add_request_handler(AddTodoHandler())
    sb.add_request_handler(LaunchRequestHandler())
    sb.add_exception_handler(CatchAllExceptionHandler())
    return sb
