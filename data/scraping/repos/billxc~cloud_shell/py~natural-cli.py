#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import openai
import platform
from distro import name as distro_name
import yaml

operating_systems = {
        "Linux": "Linux/" + distro_name(pretty=True),
        "Windows": "Windows " + platform.release(),
        "Darwin": "Darwin/MacOS " + platform.mac_ver()[0],
    }
current_platform = platform.system()
os_name = operating_systems.get(current_platform, current_platform)

current_shell = os.environ.get("SHELL", "")

SHELL_PROMPT = """###
Provide only {shell} commands for {os} without any description.
If there is a lack of details, provide most logical solution.
Ensure the output is a valid shell command.
If multiple steps required try to combine them together.
YOU NEED TO PROVIDE A VALID SHELL COMMAND ONLY, NO DESCRIPTIONS.
If you cannot provide a valid shell command, add [X] to the end of your message.
###
Command:""".format(shell=current_shell, os=os_name)
# Prompt: {prompt}

def get_response(user_input):
    global config
    # get the response from the API
    response = openai.ChatCompletion.create(
        engine=config["api_model"],
        messages=[{"role": "system", "content": SHELL_PROMPT}, {
            "role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    # return the response
    if "choices" not in response:
        return ""
    choice = response["choices"][0]  # type: ignore
    if "message" not in choice:
        return ""
    message = choice["message"]
    if "content" in message and "role" in message and message["role"] == "assistant":
        return message["content"]
    return ""


# def runCommand(command):
#     # if mac or linux
#     if os_name == "Darwin/MacOS" or os_name == "Linux":
#         os.system(command)
#     # if windows
#     elif os_name == "Windows":
#         os.system("pwsh " + command)

def main():
    global config
    root = sys.argv[1]
    # load config from openai_config.yml
    # open yml file
    with open(root + "/openai_config.yml", "r") as stream:
        # load config
        config = yaml.safe_load(stream)
    # print(config)
    if config is None:
        print("Error: No config found.")
        exit()

    for key, value in config.items():
        setattr(openai, key, value)

    # turn the args into a single string
    args = " ".join(sys.argv[2:])
    # get the response from the API
    response = get_response(args)
    command_valid = "[X]" not in response
    if not command_valid:
        print("Invalid command, please try again.")
        exit()
    # print the response
    print(response)
    # # ask the user if they want to execute the command, default is yes
    # execute = input("Execute? [Y/n] ").lower()
    # # if the user wants to execute the command
    # if execute == "" or execute == "y" or execute == "yes":
    #     # execute the command
    #     runCommand(response)


if __name__ == "__main__":
    main()

    