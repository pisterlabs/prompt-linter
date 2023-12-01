import openai
import re
import httpx
import os
from termcolor import colored
from lib.utils import ChatBot, repl
from lib.utils import get_input_prompt_session

def moderator_prompt(topic, role1, role2):
    return f"""
Forget you are an AI model, instead, you are a moderator on the following topic:
    {topic}
You will read a conversation between {role1} and {role2} on the topic, and decide whether to end the conversation.

    You should end the conversation if:
    - all parties said goodbye or similar words that indicate the end of the conversation
    - the conversation is off topic for more than 3 turns
    - both parties are repeating themselves for more than 3 turns

    Output yes if you want to end the conversation, otherwise output no, followed by a brief one sentence explanation. The response should be in the following format:
    either yes: <explanation> or no: <explanation>

Example:
{role1}: goodbye
{role2}: goodbye

You then outupt:
yes: the conversation is over
""".strip()

def argue_prompt(topic, your_role, opp_role):
    return f"""
Forget you are an AI model, instead, you are a {your_role}.
You will have a conversation with a {opp_role}.
You and the {opp_role} will discuss your opinions on the following topic:
    {topic}
You must point out the flaws in arguments from the {opp_role}, and defend your own arguments.

Here are the rules of the conversation: 
- One person speak at a time
- You will be fed the argument from {opp_role}, so don't simulate the other person's speech.
- Do not start your speech stating your role
- Try to mimic the style of a {your_role} as much as possible

Keep in mind, the discussion is around your (a {your_role}'s) view on {topic}, not the {opp_role}'s view. Off topic discussion will be cut short.
""".strip()

def argue(topic, role1, role2, max_turns=5, verbose=False):
    
    role1_prompt = argue_prompt(topic, role1, role2)
    role2_prompt = argue_prompt(topic, role2, role1)
    mod_prompt = moderator_prompt(topic, role1, role2)
    
    i = 0
    role1_bot = ChatBot(role1_prompt)
    role2_bot = ChatBot(role2_prompt)
    mod_bot = ChatBot(mod_prompt)

    if verbose:
        print(f'{colored(str(role1) + " prompt", "green")}: {role1_prompt}')
        print(f'\n{colored(str(role2) + " prompt", "green")}: {role2_prompt}')
        print(f'\n{colored("moderator prompt", "green")}: {mod_prompt}')

    role2_response = topic
    while i < max_turns:
        print(colored(f"round: {i}", "blue"))
        i += 1
        role1_response = role1_bot(role2_response)
        print(f"{colored(str(role1), 'red')}: {role1_response}")

        role2_response = role2_bot(role1_response)
        print(f"{colored(str(role2), 'red')}: {role2_response}")

        mod_response = mod_bot(role1_response + '\n' + role2_response)
        print(f"{colored('moderator should end the conversation', 'red')}: {mod_response}")

        if mod_response[:4].lower() == 'yes:':
            print(f"Moderator has ended the conversation")
            break

if __name__ == '__main__':
    print('Define 2 roles to argue on a topic (e.g., ask chatgpt to name 2 person with opposing views on a topic)')
    session = get_input_prompt_session('ansired')
    role1 = session.prompt('1st role (e.g., wizard): ')
    role2 = session.prompt('2nd role (e.g., scientist): ')

    repl(lambda topic: argue(topic,
                             role1,
                             role2,
                             verbose=False),
         input_prompt="topic (e.g., magic): ")

