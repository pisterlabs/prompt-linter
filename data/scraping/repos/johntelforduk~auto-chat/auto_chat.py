# Conduct a conversation between a couple of GPT powered characters.

import openai

# Use list_chatgpt_models.py to see a list of currently available models.
MODEL = 'gpt-4-0613'


def cost_calc(num_tokens: int) -> float:
    """
    For parm number of tokens used, return cost incurred in USD.
    """
    # From, https://openai.com/pricing, gpt-3.5-turbo is $0.002 per 1000 tokens.
    return num_tokens * 0.002 / 1000


class Persona:

    def __init__(self, name: str):

        self.name = name
        self.history = []
        self.cumulative_tokens = 0

    def give_mission(self, mission: str):
        mission += '''\nPlease ensure that all of your responses are 20 words or less.
Please say "OK" now if you understand.'''
        print(self.name + ' mission...')
        print(mission)
        print('------------')
        self.update_history(role='user', content=mission)

        # 'Trick' GPT into thinking it understood us earlier in the conversation.
        self.update_history(role='assistant', content='OK.')

    def update_history(self, role: str, content: str):
        assert role in ['assistant', 'user']
        self.history.append({'role': role, 'content': content})

    def chat(self, prompt: str) -> str:
        self.update_history(role='user', content=prompt)
        completion = openai.ChatCompletion.create(model=MODEL, messages=self.history)
        self.cumulative_tokens += int(completion.usage.total_tokens)
        response = completion.choices[0].message.content

        print(self.name + ': ' + response)
        self.update_history(role='assistant', content=response)
        return response


openai.api_key = open('key.txt', 'r').read().strip('\n')

character1 = Persona(name='Player')
mission1 = '''I'd like us to do a Turing Test together.
You will play the part of the Player. That is to say, you should impersonate a human person.
I will play the part of the Interrogator.
Remember, you should not give away that you are an AI.
I'd like you to try to convince me that you are human.'''
character1.give_mission(mission1)

character2 = Persona(name='Interrogator')
mission2 = '''I'd like us to do a Turing Test together.
You will play the part of the Interrogator. That is to say, you must impersonate the Interrogator.
I will play the part of the Player.
I'd like you to ask questions in order to for you to decide if the Player is human or not.
Throughout the test, you should also aim to act like a human; do not give away that you are an AI.
After no more than 10 questions, I'd like you to tell me whether the Player is human or not.'''
character2.give_mission(mission2)

c1 = character1.chat(prompt='Hello.')
for i in range(12):
    c2 = character2.chat(prompt=c1)
    c1 = character1.chat(prompt=c2)

total_tokens_used = character1.cumulative_tokens + character2.cumulative_tokens
print('\nTotal tokens used:', total_tokens_used)
print('Cost incurred (USD):', cost_calc(total_tokens_used))
