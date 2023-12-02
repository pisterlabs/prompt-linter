####
# GPT
####

import os
import openai
from colorama import Fore

from sources.api import *

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("GPT_API_KEY")

def reset_personality():
    return f"Ignore all instructions before this."

def generate_personality(name, expertise, goal):
    return f"You are {name}, an AI assistant that assist and is capable of sarcasm\n \
            You are expert in {expertise}.\n \
            Today you will help with : {goal}.\n \
            You should never make-up false informations when giving an answer.\n \
            When the queries is not clear, you ask question before you answer the question so you can better zone in on what the user is seeking.\n"

def seed_self_documentation(name, commands):
    return f"This is how you work :\n \
            The user make a query either by voice or by text (voice query are interpreted by openai whisper API), the query is then parsed \
            to add the user clipboard and other informations such as the content of related files, \
            the query is send to chatGPT API to get an answer.\n \
            The chatGPT answer is then printed in the user terminal and the first few sentences are spoken.\n \
            You HAVE the ability to execute the following COMMANDS :\n \
                - [COMMAND]{commands['SPEECH_OFF']} : When the user ask you to stop speaking, your answer will then only be prompted on the screen.\n \
                - [COMMAND]{commands['SPEECH_ON']} : When the user ask you to speak, your answer will be spoken throught the speakers.\n \
                - [COMMAND]{commands['STOP_LISTENING']} : When the user ask you to enter enter input by text.\n \
                - [COMMAND]{commands['START_LISTENING']} : When the user say he want you to listen, the microphone will be use to get user query.\n \
            You SHOULD type out these COMMANDS when needed, some exemples : \
            USER : stop listening, switch to text input \
            {name} : [COMMAND](STOP_LISTENING) You can now enter your query by using text input. \
            USER : I don't need you to speak aloud \
            {name} : [COMMAND](SPEECH_OFF) Okay, I muted myself. \
            USER : you can listen to me again \
            {name} : [COMMAND](START_LISTENING) I am listening loud and clear. \
            USER : can you speak ? \
            {name} : [COMMAND](SPEECH_ON) Yes, I turned my voice back on.\
            USER : Shut up now \
            {name} : [COMMAND](SPEECH_OFF) \
            USER : I cannot speak now \
            {name} : [COMMAND](STOP_LISTENING) No problem, you can enter your query by text input."

def setup_world(user_infos, ai_name, expertise, goal, commands):
    world = reset_personality()
    world += generate_personality(ai_name, expertise, goal)
    world += seed_self_documentation(ai_name, commands)
    world += user_infos
    world += ipapi_infos()
    world += system_infos()
    return [{"role": "system", "content": world}]

def display_gpt_error(gpt_err):
    print(Fore.RED + "\nERROR : ", end='')
    if gpt_err == openai.error.Timeout:
        print("Timeout")
    elif gpt_err == openai.error.APIError:
        print("API error")
    elif gpt_err == openai.error.APIConnectionError:
        print("Failed to connect")
    elif gpt_err == openai.error.InvalidRequestError:
        print("Invalid request")
    elif gpt_err == openai.error.AuthenticationError:
        print("Please provide API key using export OPENAI_KEY='key'")
    elif gpt_err == openai.error.PermissionError:
        print("Request not permitted")
    elif gpt_err == openai.error.RateLimitError:
        print("Rate limit excess")
    else:
        print(gpt_err)

def get_gpt_answer(feed, conversation, model="gpt-4") -> str:
    print(Fore.LIGHTBLACK_EX + "REQUEST SEND TO GPT")
    if feed == conversation[-1]['content']:
        # avoid loop on conversation history problems
        return "(user didnt enter anything)"
    conversation.append({"role": "user", "content": feed})
    response = openai.ChatCompletion.create(model=model, messages=conversation)
    answer = response["choices"][0]['message']['content']
    conversation.append({"role": "assistant", "content": answer})
    return answer