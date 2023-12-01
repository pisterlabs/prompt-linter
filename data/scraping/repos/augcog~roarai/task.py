import openai
from dotenv import load_dotenv
import os
load_dotenv()

def wizard_coder(history: list[dict]):
    DEFAULT_SYSTEM_PROMPT = history[0]['content']+'\n\n'
    B_INST, E_INST = "### Instruction:\n", "\n\n### Response:\n"
    messages = history.copy()
    messages_list=[DEFAULT_SYSTEM_PROMPT]
    messages_list.extend([
        f"{B_INST}{(prompt['content']).strip()}{E_INST}{(answer['content']).strip()}\n\n"
        for prompt, answer in zip(messages[1::2], messages[2::2])
    ])
    messages_list.append(f"{B_INST}{(messages[-1]['content']).strip()}{E_INST}")
    return "".join(messages_list)

def gpt(history: list[dict]):
    l=[x['content'] for x in history]
    return '\n---\n'.join(l)

model='local'
# model='openai'
if model=='local':
    openai.api_key = "empty"
    openai.api_base = "http://localhost:8000/v1"
elif model=='openai':
    openai.api_key = os.getenv("OPENAI_API_KEY")

human_message = '''
What is the purpose of the kinematics_solver parameter in the `kinematics.yaml` file, and what should it be replaced with to utilize the `MoveItOPWKinematicsPlugin?`
'''
# human_message='who are you'



# system_message='you are a bot'
def bullet_point_creator(m):
    # system_message = 'Your task is to summarize passage given by the user into bullet points'
    # system_message = 'Given the content and the document_hierarchy_path of a document, describe the detailed questions you can ask based on its content.'
    system_message = 'Generate the answer to this question'
    messages=[{"role": "system", "content": system_message}, {"role": "user", "content": human_message}]
    # if model=='local':
    #     prompt=wizard_coder(history)
    # elif model=='openai':
    #     prompt=gpt(history)
    # print(prompt)
    if model=='local':
        prompt=wizard_coder(messages)
        completion = openai.Completion.create(
            model=m, prompt=prompt, temperature=0
        )
    elif model=='openai':
        prompt=gpt(messages)
        completion = openai.ChatCompletion.create(
            model=m, messages=messages, temperature=0
        )
    print(completion)

    bullet=completion['choices'][0]['message']["content"]
    print(bullet)


bullet_point_creator('gpt-3.5-turbo')
# test_chat_completion('gpt-4')
