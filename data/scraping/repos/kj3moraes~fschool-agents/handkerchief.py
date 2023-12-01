import asyncio
import os
from copy import deepcopy
import openai
import tiktoken

openai.api_key = os.environ['OPENAI_API_KEY']


class Handkerchief():
    def __init__(self):
        self.chunks = []
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

    def index(self, text):
        tokens = self.tokenizer.encode(text)
        while len(tokens) > 0:
            chunk = self.tokenizer.decode(tokens[:15000])
            self.chunks.append(chunk)
            tokens = tokens[15000:]

    async def sneeze(self, messages, **openai_kwargs):
        async def _search():
            results = []
            for chunk in self.chunks:
                results.append(self.retrieve(messages, chunk))
            return await asyncio.gather(*results)
        
        results = await _search()
        results = [result.choices[0].message.content for result in results]
        return self.generate(messages, results, **openai_kwargs)

    async def retrieve(self, messages, chunk):
        messages = deepcopy(messages)
        system_prompt = (
            "You find information in a block of text before responding to the user.\n"
            "If there is not relevant information in the text, tell the user immediately.\n"
            "List all of the information that might be relevant to the user.\n"
            "List exact quotes from the text to support your answer."
        )
        if messages[0]['role'] == 'system':
            messages[0]['content'] = system_prompt
        else:
            messages.insert(0, {'role': 'system', 'content': system_prompt})

        user_message = (
            "Below is a file I found that might help you with your response. "
            "Try to find some relevant information in it:\n\n" + str(chunk)
        )
        messages.append({'role': 'user', 'content': user_message})

        return await openai.ChatCompletion.acreate(messages=messages, model='gpt-3.5-turbo-16k')
    
    def generate(self, messages, results, **openai_kwargs):
        messages = deepcopy(messages)
        
        system_prompt = "Use the information you found in the file to respond to the user."
        if messages[0]['role'] == 'system':
            messages[0]['content'] += '\n' + system_prompt
        else:
            messages.insert(0, {'role': 'system', 'content': system_prompt})

        user_message = "Here's some relevant information I found in your files:\n\n" + "\n\n---\n\n".join(results)
        messages.append({'role': 'user', 'content': user_message})

        return openai.ChatCompletion.create(messages=messages, **openai_kwargs)


def test(message):
    handkerchief = Handkerchief()

    with open('globe.txt', 'r') as f:
        text = f.read()

    handkerchief.index(text)
    print(f'Split text into {len(handkerchief.chunks):d} chunks')
    
    messages = [{'role': 'user', 'content': message}]
    response = handkerchief.sneeze(messages, model='gpt-4', stream=True)

    print()
    for chunk in response:
        choice = chunk.choices[0]
        if not choice.delta:
            continue
        text = choice.delta.content
        print(text, end='', flush=True)
    print()


if __name__ == '__main__':
    test("When was the first globe made?")