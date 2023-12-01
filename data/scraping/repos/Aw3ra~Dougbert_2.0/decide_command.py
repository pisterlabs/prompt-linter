from openAI.functions import generate_text
import os
import json
from dotenv import load_dotenv



# from general_functions import find_json



# ---------------------------------------------------------------------------------#
# Function to decide which command to run based on the tweet
# Inputs:   tweet - an array of dictionaries in this format:
#                   {'role': 'user', 'content': 'Hello, my name is Doug'}
#                   {'role': 'assistant', 'content': 'Hello Doug, how are you doing today?'}
# Outputs:  the command to run
# ---------------------------------------------------------------------------------#
def decide_command(conversation):
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        twitter_handle = str(os.getenv("TWITTER_HANDLE"))
        prompts = json.load(open('src/commands/command_generating_prompts.json', 'r'))['command_decision']
        with open('src/profile.json', 'r', encoding='utf-8') as f:
            LIST_OF_COMMANDS = json.load(f)['config_data']['command_status']
            # Remove the commands that have a status of False
            LIST_OF_COMMANDS = [command for command in LIST_OF_COMMANDS if LIST_OF_COMMANDS[command] == True]
        # prompts = find_json.find_json_file('command_generating_prompts.json')['command_decision']

        command_string = ""
        for command in LIST_OF_COMMANDS:
            command_string += command + "\n"
            # Get the last two messages
            conversation = conversation[-1:]
        
        system_prompt = {"role": "system", "content": prompts["system_prompt"].replace("{COMMAND_LIST}", command_string).replace("{USER_NAME}", twitter_handle)}
        examples = []
        all_examples = prompts['examples']
        for example in all_examples:
            if example.startswith("example_"):
                examples.append({"role": "user", "content": all_examples[example].replace("{USER_NAME}", twitter_handle)})
            elif example.startswith("answer_"):
                examples.append({"role": "assistant", "content": all_examples[example]})

        conversation.insert(0, system_prompt)
        for example in examples:
            conversation.insert(1, example)
        # -------------------------------------------------------------------------#
        # Get the text to generate
        # -------------------------------------------------------------------------#
        return generate_text.generate_text(api_key, prompt = conversation)
    except Exception as e:
        raise e

if __name__ == '__main__':
    print(decide_command([{'role': 'user', 'content': 'Hello, my name is Doug'}]))
