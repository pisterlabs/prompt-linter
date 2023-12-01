import os
from openai import OpenAI

class MyOpenAI:
    def __init__(self, api_key, model='gpt-4-1106-preview', temprature=1, top_p=1):
        self.model = model
        self.temprature = temprature
        self.top_p = top_p
        self.api_key = api_key
        self.client = OpenAI()

    def complete(self, messages):
        if type(messages) == str: # In case someone accidentally passes in a string instead of a list of messages
            messages = [{"role": "user", "content": messages}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={'type': "text"},
            temperature=self.temprature,
        )
        if resp.choices[0].finish_reason != 'stop':
            raise Exception(f"OpenAI chat completion failed with reason: {resp['choices'][0]['finish_reason']}")
        return resp, resp.choices[0].message.content
        
    def json_mode_complete(self, messages, tools, tool_choice='auto'):
        '''
            System message must contain the string 'JSON' in it.
        '''
        if type(messages) == str: # In case someone accidentally passes in a string instead of a list of messages
            messages = [{"role": "user", "content": messages}]
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={'type': "json_object"},
            messages=messages,
            temperature=self.temprature,
            tools=tools,
            tool_choice=tool_choice
        )
        if resp.choices[0].finish_reason != 'stop':
            raise Exception(f"OpenAI chat completion failed with reason: {resp['choices'][0]['finish_reason']}")
        return resp, resp.choices[0].message.tool_calls[0].function.arguments