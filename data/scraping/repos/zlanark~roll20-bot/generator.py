import openai
import os
import math

from messages import *
from messages import Message
from exceptions import APIError

class Generator():
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    @staticmethod
    def _is_continuation(msg1: Message, msg2: Message):
        if(msg1.has_tag(MessageTag.CHARACTER) and msg2.has_tag(MessageTag.CHARACTER) \
           and msg1.get_character() == msg2.get_character()):
            return True
        return False

    @staticmethod
    def _format_messages(messages: list[Message], for_character: str|None) -> str:
        output = ""
        for index, msg in enumerate(messages):
            if(msg.has_tag(MessageTag.CHARACTER)):
                continued = (__class__._is_continuation(messages[index-1], msg) if index != 0 else False)
                continues = (__class__._is_continuation(msg, messages[index+1]) if index != len(messages)-1 else False)

                if(not continued):
                    output += msg.get_character() + ': \"' # type: ignore
                output += msg.get_content().removeprefix('"').removesuffix('"')

                if(continues):
                    output += '\n'
                else:
                    output += '"\n\n'
            else:
                output += msg.get_content() + "\n\n"
        if(for_character != None):
            output += f'{for_character}: '
        return output
            
    # Filtering out non-narration/non-character messages should be done prior to calling this method
    @staticmethod
    def get_response(history: list[Message], system_prompt, as_character: str, model='gpt-4'):
        input = __class__._format_messages(messages=history, for_character=as_character)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        try:
            response = openai.ChatCompletion.create(
                model=model.lower(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input},
                ]
            )
            return response
        except Exception as error:
            raise APIError

    @staticmethod
    def count_tokens(char_count: int) -> int:
        #This is only an approximation
        return math.ceil(0.25*(char_count))