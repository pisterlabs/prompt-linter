#!/bin/python3

import os
import openai
import argparse
from datetime import datetime
import subprocess

import message_history
from model_handling import (
    extract_model_name,
    uses_legacy_completions,
    GPT_3_MODEL_NAME,
    GPT_4_32K_MODEL_NAME,
    GPT_4_BASE_MODEL_NAME,
    GPT_4_MODEL_NAME,
    DEFAULT_MODEL_NAME,
)


DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate AI assistant who provides BRIEF excellent responses to queries (NEVER say fluff like 'As an AI')."

DEFAULT_FILENAME = "GPT_ATTACHED_CONTEXT.txt"


def get_org_id(short_model_name: str) -> str:
    """
    Check if model-specific organization ID is saved as environment variable.
    For example, if $OPENAI_ORGANIZATION_32k is saved, uses that as the org id.
    If no such org id found, defaults back to $OPENAI_ORGANIZATION.
    """
    org_id_varname = f"OPENAI_ORGANIZATION_{short_model_name}"
    org_id = os.getenv(org_id_varname)
    if not org_id:
        org_id = os.getenv("OPENAI_ORGANIZATION")
    return org_id


if __name__ == "__main__":
    # Parse command line input

    # Set up parser
    parser = argparse.ArgumentParser(
        description="Simple command line interface for OpenAI API."
    )
    parser.add_argument(
        "prompt", nargs="?", type=str, help='Your prompt (e.g. "Hello World").'
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display entire usage history or (if conversation ID given) a specific conversation's history",
    )
    parser.add_argument(
        "-r",
        "--reply",
        action="store_true",
        help="Reply to previous conversation (default is most recent, or select with conversation ID)",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs=1,
        default="3"
        if DEFAULT_MODEL_NAME == GPT_3_MODEL_NAME
        else "4"
        if DEFAULT_MODEL_NAME == GPT_4_MODEL_NAME
        else "",
        type=str,
        help="Model to use ('3' for gpt-3.5-turbo, '4' for gpt-4-0314, '32k' for gpt-4-32k, 'base' for gpt-4-base)",
    )
    parser.add_argument(
        "-c",
        "--conversation_id",
        type=int,
        help="Specify which conversation to reply to or display",
    )
    parser.add_argument(
        "-s",
        "--system",
        nargs="?",
        default=DEFAULT_SYSTEM_PROMPT,
        type=str,
        help=f'System prompt to use. Default: "{DEFAULT_SYSTEM_PROMPT}"',
    )
    parser.add_argument(
        "-f",
        "--fileread",
        action="store_true",
        help=f"If selected, appends to prompt content from {DEFAULT_FILENAME}",
    )
    parser.add_argument(
        "-w",
        "--filewrite",
        action="store_true",
        help=f"If selected, opens {DEFAULT_FILENAME} in vim to let you paste content in.",
    )

    parser.add_argument(
        "-t", "--temperature", help=f"Set the temperature for the query.", type=float
    )

    # Parse and extract args
    args = parser.parse_args()

    user_prompt = args.prompt
    reply_mode = args.reply
    display_mode = args.display
    short_model_name = args.model[0]
    model_name = extract_model_name(short_model_name)
    conv_id = args.conversation_id
    system_prompt = args.system
    fileread = args.fileread
    filewrite = args.filewrite
    temperature = args.temperature

    # First handle display mode
    if display_mode:
        if conv_id != None:
            message_history.display_history(conv_id)
        else:
            message_history.display_all_history()
        exit(0)

    # Next handle write mode
    if filewrite:
        subprocess.run(["vim", DEFAULT_FILENAME])
        exit(0)

    # Otherwise enter conversation mode

    if user_prompt == None:
        parser.print_help()
        exit(1)

    # If history list does not exist, create it
    # (i.e. there is no message_history.pkl saved where you expect)
    if not message_history.is_history_list():
        message_history.init_history_list()

    chat_names = message_history.get_chat_names()
    history_list = message_history.get_history_list()

    # Get current chat name and history
    if reply_mode:
        # If conv_id specified, reply to that, otherwise reply to most recent conversation
        if conv_id != None:
            reply_index = conv_id
        else:
            reply_index = -1
        try:
            current_history = history_list[reply_index]
            current_chat_name = current_history.get_chat_name()
        except IndexError:
            print("Can't reply to empty history.")
    else:
        current_chat_name = str(len(chat_names))
        current_history = message_history.History(
            current_chat_name, system_prompt, uses_legacy_completions(model_name)
        )

    if fileread:
        try:
            with open(DEFAULT_FILENAME, "r") as f:
                fileread_content = f.read()
        except FileNotFoundError:
            print(
                f"Error: {DEFAULT_FILENAME} not found. First run `gpt -w` to write to this file."
            )
            exit(1)
        full_prompt = user_prompt + fileread_content
    else:
        full_prompt = user_prompt

    current_history.append_user_message(full_prompt)

    # Talk to model

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = get_org_id(short_model_name)

    try:
        optional_args = (
            {"temperature": temperature} if temperature is not None else dict()
        )

        if uses_legacy_completions(model_name):
            prompt = current_history.get_message_history(
                for_openai=True,
                legacy=True,
            )
            print(prompt)
            completion = openai.Completion.create(
                model=model_name,
                prompt=prompt,
                stream=True,
                max_tokens=4000,
                **optional_args,
            )
            response = ""
            for chunk in completion:
                chunk_message_str = chunk["choices"][0]["text"]
                response += chunk_message_str
                print(chunk_message_str, end="", flush=True)

        else:
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=current_history.get_message_history(for_openai=True),
                stream=True,
                **optional_args,
            )

            response = ""
            for chunk in completion:
                chunk_message = chunk["choices"][0]["delta"]
                chunk_message_str = chunk_message.get("content", "")
                response += chunk_message_str
                print(chunk_message_str, end="", flush=True)

    except KeyboardInterrupt:
        print("<KeyboardInterrupt>", flush=True)
    else:
        print()

    # Log to history
    if reply_mode:
        message_history.update_history(reply_index, user_prompt, response, model_name)
    else:
        # Add GPT's response to current_history object
        current_history.append_response(response, model_name)
        # Save current_history object to message history
        message_history.append_history(current_history)

    # Log all raw data

    message_history.write_to_log(
        f"time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
    )
    message_history.write_to_log(f"prompt: {user_prompt}\n")
    message_history.write_to_log(str(completion) + "\n\n")
