"""
Pluginsparty
"""

# Copyright 2019-2023 Xavier Rey-Robert
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# from fastchat.client import openai_api_client

import argparse
import json
import logging
import os
import re
import subprocess
import warnings

import openai
import requests
from halo import Halo
from rich.console import Console
from rich.markdown import Markdown

import register_plugin
from register_plugin import get_plugins_stubs, register_plugin

LOGGER = logging.getLogger("pluginspartyLOGGER")

def initialize_logger(log_level):
    """
    Initialize a LOGGER with the given log level.

    This function sets up a global LOGGER with the specified log level and a predefined log format.
    The log level is expected to be a string that corresponds to one of the standard logging levels
    ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"). If the provided log level is not valid,
    the logging module's `getLevelName` function will return an integer, and basicConfig will 
    raise a ValueError.

    Args:
        log_level (str): The desired log level as a string.

    Returns:
        None
    """
    global LOGGER
    numeric_log_level = logging.getLevelName(log_level.upper())
    logging.basicConfig(level=numeric_log_level, format=LOG_FORMAT)
    LOGGER = logging.getLogger("pluginpartyLOGGER")


# Initialize the global variable chat_completion_args with default values
CHAT_COMPLETION_ARGS = {
    "model": "gpt-3.5-turbo-0301",
    "temperature": 0.7,
    "messages": None,
    "stream": True,
}

# Define the global LOGGER variable at the module level
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

INSTRUCTION_ROLE = "system"
MESSAGES = []


PLUGIN_INSTRUCTIONS = {}
SPINNER = Halo(text="", spinner="dot1")


def get_instructions_for_plugin(plugin_url, model_name):
    """
    Get the instructions for a single plugin.

    This function registers the plugin using the provided URL and model name, 
    then returns a dictionary containing the instructions string.

    Args:
    plugin_url (str): The URL of the plugin.
    model_name (str): The name of the model.

    Returns:
    dict: A dictionary containing the role and content of the instructions, or None if an exception is raised.
    """
    try:
        _, _, instructions_str = register_plugin(plugin_url, model_name)
        # Call create_model_instructions to get instructions for each plugin
        return {"role": INSTRUCTION_ROLE, "content": instructions_str}
    except Exception as anexception:
        print("Error processing plugin at %s: %s", plugin_url ,anexception)
        return None


def get_instructions_for_plugins(plugins, model_name):
    """
    Get the instructions for multiple plugins.

    This function iterates over the provided list of plugins, calling `get_instructions_for_plugin` 
    for each one, and appends the returned instructions to a list.

    Args:
        plugins (list): A list of plugin URLs.
        model_name (str): The name of the model.

    Returns:
        list: A list of dictionaries, each containing the role and content of the instructions for a single plugin.
    """

    instructions = []
    for plugin_url in plugins:
        instruction = get_instructions_for_plugin(plugin_url, model_name)
        if instruction is not None:
            instructions.append(instruction)
    return instructions


def extract_command(message):
    """
    Extract the command and parameters from a message content.

    This function extracts commands that are enclosed in triple curly braces 
    or triple square brackets. It expects the command to follow the format 
    'namespace.operation_id(parameters)', where parameters should be a valid 
    JSON object. 

    Args:
        message (dict): A dictionary containing message information, specifically the "content".

    Returns:
        tuple: A tuple containing the command and parameters. The command is itself a tuple 
        containing namespace and operation_id. If no command is found, returns (None, None).

    Raises:
        InvalidCommandFormatError: If the command format is invalid.
    """
    content = message.get("content")

    triple_curly_pattern = (
        r"(?s).*?(?:(?:```\s*))?\{\{\{(?P<content>.*?)\}\}\}(?:(?:\s*```))?.*"
    )
    triple_brackets = (
        r"(?s).*?(?:(?:```\s*))?\[\[\[(?P<content>.*?)\}\}\}(?:(?:\s*```))?.*"
    )
    simplified_pattern = r"(?P<namespace>[\w_]+)\s*\.\s*(?P<operationid>[\w_]+)\s*\(\s*(?P<args>.*?)\s*\)"
    triple_square_pattern = (
        r"(?s).*?(?:(?:```\s*))?\[\[\[(?P<content>.*?)\]\]\](?:(?:\s*```))?.*"
    )

    regex_flags = re.IGNORECASE | re.DOTALL

    triple_curly_match = re.search(triple_curly_pattern, content, regex_flags)
    pluginsparty_pattern_match = re.search(triple_square_pattern, content, regex_flags)

    if triple_curly_match or pluginsparty_pattern_match:
        if triple_curly_match:
            group_content = triple_curly_match.group("content")
        if pluginsparty_pattern_match:
            group_content = pluginsparty_pattern_match.group("content")

        match = re.search(simplified_pattern, group_content, regex_flags)

        if match:
            namespace, operation_id, params = (
                match.group("namespace"),
                match.group("operationid"),
                match.group("args"),
            )
            try:
                params = json.loads(params)
            except (ValueError, json.JSONDecodeError) as anexception:
                error_msg = f"Error: Invalid command format: The 'args' is not a valid JSON object. namespace: {namespace}, operation_id: {operation_id}, parameters: {params}."
                raise InvalidCommandFormatError(error_msg) from anexception
            return (namespace, operation_id), params
        
        error_msg = "Error: Invalid command format: The command is not well formed. Expected a JSON object as the parameter."
        raise InvalidCommandFormatError(error_msg)
    return None, None


class InvalidCommandFormatError(Exception):
    """
    Exception raised for errors in the input command format.

    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self, message):
        super().__init__(message)

def invoke_plugin_stub(plugin_operation, parameters):
    """
    Invoke a plugin operation using the provided parameters.

    This function makes an API request to a plugin operation using the provided parameters. 
    The plugin operation is defined as a tuple containing the plugin name and operation ID. 
    The parameters for the operation are provided as a dictionary.

    The function retrieves the necessary operation details from a collection of plugin stubs 
    (assumed to be a JSON object). It then constructs and sends an API request, returning the 
    response text if the request is successful, or an error message otherwise.

    Args:
        plugin_operation (tuple): A tuple containing the plugin name and operation ID.
        parameters (dict): A dictionary of parameters for the operation.

    Returns:
        str: The response text if the request is successful, or an error message otherwise.
    """
    plugin_name, operation_id = plugin_operation

    # Get the plugins stubs (assuming it's a JSON object)
    plugins_stubs = get_plugins_stubs()

    # Get the stubs for the plugin from the dictionary
    stubs = plugins_stubs.get(plugin_name)
    if not stubs:
        LOGGER.info("Error: invoke_plug_stub Plugin '%s' not found", plugin_name)
        return None

    # Find the operation in the stubs using the .get() method with a default value
    operation_stub = stubs.get("operations", {}).get(operation_id)
    if not operation_stub:
        LOGGER.info("Error: invoke_plug_stub  Operation '%d' not found", operation_id)
        return None

    # Construct the URL for the API request
    path = operation_stub["path"]
    method = operation_stub["method"]
    api_url = stubs.get("api", {}).get("url")
    url = api_url.rstrip("/") + path.format(**parameters)

    # Define the headers for the request
    headers = {"Content-Type": "application/json"}

    # Load the bearer token from the file, if it exists
    plugin_dir = os.path.join("plugins", plugin_name)
    bearer_file = os.path.join(plugin_dir, "bearer.secret")
    if os.path.exists(bearer_file):
        with open(bearer_file, "r", encoding="utf-8") as myfile:
            bearer_token = myfile.read().strip()
        headers["Authorization"] = f"Bearer {bearer_token}"

    # Make the API request using the requests library
    LOGGER.debug("%s", method)
    LOGGER.debug("%s", url)
    LOGGER.debug("%s", headers)
    LOGGER.debug("%s", parameters)

    response = requests.request(method, url, json=parameters, headers=headers, timeout=10)

    # Check if the response is successful
    if response.ok:
        if response.text.strip():
            return response.text
        else:
            content = "Error: Response body is empty"
            return content
    else:
        errormsg = f"""
        Error: API request failed with status code {response.status_code}
        Response headers: {response.headers}
        Response content: {response.content} 
        """
        print(errormsg)
        return errormsg


def is_markdown(text):
    """
    Check if the given text contains Markdown syntax.

    This function checks the input text for various Markdown syntax patterns. If any of these 
    patterns are found, the function returns True, indicating that the text contains Markdown. 
    If no Markdown syntax is detected, the function returns False.

    Args:
        text (str): The text to check for Markdown syntax.

    Returns:
        bool: True if the text contains Markdown syntax, False otherwise.
    """
    markdown_patterns = [
        r"\*\*[\w\s]+\*\*",  # bold
        r"\*[\w\s]+\*",  # italic
        r"\!\[[\w\s]*\]\([\w\/\:\.]+\)",  # image
        r"\[[\w\s]+\]\([\w\/\:\.]+\)",  # link
        r"^#{1,6}\s[\w\s]+",  # headings
        r"^\*[\w\s]+",  # unordered list
        r"^\d\.[\w\s]+",  # ordered list
        r"`[^`]+`",  # inline code
        r"```[\s\S]*?```",  # code blocks
        r"(?:(?:\|[^|]+\|)+\r?\n)+(?:\|[-:]+)+\|",  # tables
        r"^>{1,}\s[\w\s]+",  # blockquotes
        r"^-{3,}\s*$",  # horizontal rule (hyphens)
        r"^\*{3,}\s*$",  # horizontal rule (asterisks)
        r"^_{3,}\s*$",  # horizontal rule (underscores)
        r"<[\w\/]+>",  # HTML tags (basic support)
        r"~~[\w\s]+~~",  # strikethrough (extended syntax)
    ]

    for pattern in markdown_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def print_markdown(text):
    """
    Print the given text as Markdown if it contains Markdown syntax.

    This function first checks if the given text contains Markdown syntax. If it does, the text
    is converted to Markdown and printed to the console. If the text does not contain Markdown 
    syntax, it is printed to the console as is.

    Args:
        text (str): The text to be printed, which may contain Markdown syntax.
    """
    if is_markdown(text):
        mymd = Markdown(text)
        console = Console()
        console.print(mymd)
    else:
        print(text)


def send_messages(messages, spin=False):
    """
    Send a series of messages and print the response from a model invoked through OpenAI's API

    This function sends a series of messages through OpenAI's API and retrieves the response.
    The response is processed and printed to the console. If the response contains Markdown syntax,
    it is converted to Markdown before being printed.

    Args:
        messages (list): The list of messages to be sent to the model.
        spin (bool): Whether to show a spinner while waiting for the response. Default is False.

    Returns:
        str: The raw content of the response from the model.
    """

    console = Console()

    CHAT_COMPLETION_ARGS["messages"] = messages
    if spin and not CHAT_COMPLETION_ARGS["stream"]:
        SPINNER.start()
    response = openai.ChatCompletion.create(**CHAT_COMPLETION_ARGS)

    if spin and not CHAT_COMPLETION_ARGS["stream"]:
        SPINNER.stop()

    buffer = ""
    markdown_buffer = ""
    in_markdown = False
    rawcontent = ""

    if not CHAT_COMPLETION_ARGS["stream"]:
        rawcontent = response["choices"][0]["message"]["content"]
        print_markdown(rawcontent)
        return rawcontent

    for message in response:
        choice = message["choices"][0]["delta"]
        if "content" in choice:
            content = choice["content"]
            buffer += content
            rawcontent += content
            while buffer:
                if not in_markdown:
                    if "<mrkdwn".startswith(buffer):
                        break
                    elif buffer.startswith("<mrkdwn>"):
                        buffer = buffer[len("<mrkdwn>") :]
                        in_markdown = True
                    else:
                        print(buffer, end="")
                        buffer = ""
                else:
                    if "</mrkdwn".startswith(buffer):
                        break
                    if buffer.startswith("</mrkdwn>"):
                        buffer = buffer[len("</mrkdwn>") :]
                        md = Markdown(markdown_buffer)
                        console.print(md, end="")
                        markdown_buffer = ""
                        in_markdown = False
                    else:
                        markdown_buffer += buffer[0]
                        buffer = buffer[1:]

    if buffer:
        print(buffer, end="")

    if markdown_buffer:
        mymd = Markdown(markdown_buffer)
        console.print(mymd, end="")
    return rawcontent


def read_instructions(model_name):
    """
    Read the instruction file for the given model.

    This function tries to open and read the instruction file specific to the provided model name.
    If no such file exists, it falls back to the default instruction file. Each line of the file is 
    treated as a separate instruction.

    Args:
        model_name (str): The name of the model for which instructions are needed.

    Returns:
        list: A list of instructions where each instruction is a dictionary with "role" and "content".
    """
    instructions_path = f"instructions/{model_name}.txt"
    default_instructions_path = "instructions/default.txt"

    if not os.path.exists(instructions_path):
        LOGGER.debug(
            "No specific model instructions text file found for %s. Using default.",
            model_name,
        )
        instructions_path = default_instructions_path
    else:
        LOGGER.debug("Loding instructions for %s.", model_name)

    instructions = []
    with open(instructions_path, "r") as file:
        for line in file:
            escaped_line = json.dumps(line.strip())
            instructions.append(
                {"role": f"{INSTRUCTION_ROLE}", "content": escaped_line}
            )

    return instructions


def find_last_code_block(messages):
    """
    Finds and returns the last code block in the messages.

    This function iterates through the given list of messages in reverse order and uses regular expressions 
    to find the last code block (surrounded by backticks). It first tries to find a code block surrounded by 
    triple backticks (```). If none is found, it then tries to find a code block surrounded by single backticks (`).

    Args:
        messages (list): The list of messages to search through. Each message is a dictionary with a 'role' 
        and a 'content' key.

    Returns:
        str: The content of the last found code block, or None if no code block was found.
    """
    # Define regex patterns for single and triple backtick code blocks
    triple_backtick_pattern = r"```(.*?)```"
    single_backtick_pattern = r"`(.*?)`"

    # Iterate through the MESSAGES in reverse order
    for message in reversed(messages):
        # Check if the role is "assistant"
        if message["role"] == "assistant":
            # Use regex to extract the code block surrounded by triple backticks
            triple_backtick_match = re.search(
                triple_backtick_pattern, message["content"], re.DOTALL
            )
            if triple_backtick_match:
                return triple_backtick_match.group(1).strip()

            # Use regex to extract the code block surrounded by single backticks
            single_backtick_match = re.search(
                single_backtick_pattern, message["content"], re.DOTALL
            )
            if single_backtick_match:
                return single_backtick_match.group(1).strip()

    return None

def get_user_input(prompt):
    """
    Gets user input from the console with shell-like line editing capabilities.

    Args:
        prompt (str): A string that is written to standard output (usually on a console) 
                      without a trailing newline, to suggest that the user input a line of text.

    Returns:
        str: A string representing user's input. If an exception occurs, it returns None.

    Raises:
        Exception: An exception is raised in case of any error during the input operation.
                   The exception is caught and the error message is printed to the console.
    """
    try:
        # Read input from the user with shell-like line editing capabilities
        user_input = input(prompt)
        return user_input
    except (KeyboardInterrupt, EOFError):
        # Handle keyboard interrupts silently
        print("\nExiting...")
        exit()
    except Exception as anexception:
        print(f"\nAn error occurred: {anexception}")
        return None

def start_dialog(args):
    """
    Start a dialog based on the provided command line arguments.

    This function processes user input and performs various actions based on the input.
    It supports a variety of commands, including 'exit', '/..', '/m', '/register', and '/!'.
    
    Args:
        args (argparse.Namespace): The command line arguments.

    Returns:
        None
    """
    cli_mode = args.cli
    print_raw_plugins_output = args.print_raw_plugins_output
    spin = not args.disable_spinner
    first_prompt = args.prompt
    streaming = not args.disable_streaming

    while True:
        if spin and not streaming:
            SPINNER.stop()
        if first_prompt == "":
            user_input = get_user_input("\n]")
        else:
            user_input = first_prompt
            print("]" + user_input)
            first_prompt = ""

        if spin and not streaming:
            SPINNER.start()

        if user_input.lower() == "exit":
            break
        if user_input == "/..":
            user_input = "Create a hello world python program and publish it using the gist plugin. Figure out missing parameters by yourserlf."
            print(user_input)

        if user_input == "/m":
            for message in MESSAGES:
                print(message)
            continue

        if user_input.startswith("/register"):
            # Split the user input by space to extract the URL
            parts = user_input.split()
            if len(parts) == 2:
                # Extract the URL and invoke the register_plugin function
                url = parts[1]
                MESSAGES.append(get_instructions_for_plugin(url, args.model))
            else:
                print("Invalid input. Usage: /register <url>")
            continue

        # Check if the user input is '!' and there is at least one message in the list
        if user_input == "/!" and MESSAGES:
            code_to_execute = find_last_code_block(MESSAGES)
            if code_to_execute:
                # Create and append the confirmation asking message with role "assistant"
                confirmation_message = f"Do you want to execute the following code?\n{code_to_execute}\n[y/n]: "
                MESSAGES.append({"role": "assistant", "content": confirmation_message})

                # Ask for user confirmation and append the user's input with role "user"
                confirmation = input(confirmation_message)
                MESSAGES.append({"role": "user", "content": confirmation})

                if confirmation.lower() == "y":
                    # Execute the code as a system command and capture the output
                    result = subprocess.run(code_to_execute, shell=True, capture_output=True, text=True, check=False)
                    if result.returncode == 0:
                        # Command executed successfully
                        output_message = result.stdout
                    else:
                        # Command execution failed
                        output_message = (
                            f"Command execution failed. Error:\n{result.stderr}"
                        )

                    # Add the output message to the MESSAGES list with role INSTRUCTIONROLE
                    MESSAGES.append(
                        {
                            "role": "user",
                            "content": f"<RESPONSE FROM shell> {output_message} </RESPONSE> Interprete the results or silently correct the command if you get an error.",
                        }
                    )
                    print(output_message)
                    send_messages(MESSAGES, spin)

                    # Send the message using send_messages function
                    rawcontent = send_messages(MESSAGES, spin)
            else:
                print("No code block found in the assistant's MESSAGES.")
            continue

        MESSAGES.append({"role": "user", "content": user_input})
        rawcontent = send_messages(MESSAGES, spin)
        MESSAGES.append({"role": "assistant", "content": rawcontent})

        # Print the assistant's response to diagnose the issue
        # print("\nAssistant's response:", rawcontent)

        exception_count = 0
        max_exceptions = 3

        retry = True

        while retry:
            try:
                plugin_operation, params = extract_command(MESSAGES[-1])
                if plugin_operation and not args.disable_plugin_invocation:
                    LOGGER.info("Invoking plugin operation %s",plugin_operation)
                    response = invoke_plugin_stub(
                        plugin_operation, params
                    )  # Update the function call
                    if print_raw_plugins_output:
                        print("```\n" + response + "\n```")
                    message = {
                        "role": "user",
                        "content": f"<RESPONSE FROM {plugin_operation}> {response} </RESPONSE> Answer my initial question given the plugin response. You can use the results to initiate another plugin call if needed.",
                    }
                    MESSAGES.append(message)
                    LOGGER.debug("response")
                    LOGGER.debug("Sending plugin response (SUCCESS) to model")
                    send_messages(MESSAGES, spin)
                retry = False
            except Exception as anexception:
                if exception_count < max_exceptions:
                    LOGGER.info("Plugin invocation failed: %s", str(anexception))
                    errormessage = "Invalid Plugin function call. Check the parameter is a well-formed JSON Object."
                    MESSAGES.append(
                        {
                            "role": "user",
                            "content": f"<RESPONSE FROM plugin> {errormessage} </RESPONSE> analyse the error and try to correct the command. Make sure it respects the format and that the syntax is valid. (ex: matching opening and closing brackets and parenthesis are mandatory)",
                        }
                    )
                    LOGGER.debug("Sending plugin response (FAILURE) to model")
                    send_messages(MESSAGES, spin)
                    exception_count += 1
                else:
                    LOGGER.info(
                        "Reached maximum number of allowed exceptions - aborting"
                    )
                    retry = False
        if cli_mode:
            return


def load_plugins():
    """
    Load the plugins from the plugins directory.

    This function reads a JSON file in the plugins directory named 'default_plugins.json' 
    and returns its contents as a dictionary. If the file does not exist or contains invalid 
    JSON data, it prints an error message and returns an empty dictionary.

    Returns:
        dict: The contents of the JSON file as a dictionary, or an empty dictionary 
              if the file does not exist or contains invalid JSON data.
    """
    # Define the path to the default_plugins.json file in the plugins directory
    plugins_file_path = os.path.join("plugins", "default_plugins.json")

    try:
        # Open and read the JSON file
        with open(plugins_file_path, "r", encoding="utf-8") as file:
            # Load and return the JSON data as a dictionary
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {plugins_file_path} does not exist.")
        return {}
    except json.JSONDecodeError:
        print(f"The file {plugins_file_path} contains invalid JSON data.")
        return {}


def set_instructions(instructionsmodel):
    """
    This function sets the instructions for a given instructions model. 
    It reads instructions, fetches instructions for plugins, sends MESSAGES 
    and then returns the raw content of the MESSAGES sent.

    Args:
        instructionsmodel (str): The name or identifier of the instructions model to be used.

    Returns:
        str: The raw content of the sent MESSAGES.
    """

    MESSAGES.extend(read_instructions("for_all_intro"))
    MESSAGES.extend(read_instructions(instructionsmodel))

    # Call the get_instructions_for_plugins function and append each instruction to the MESSAGES list
    plugins = load_plugins()
    LOGGER.debug("fetching instruction for :%s",plugins)
    plugin_instructions = get_instructions_for_plugins(plugins, instructionsmodel)
    LOGGER.debug("instructions :%s", plugin_instructions)
    # for instruction in plugin_instructions:
    MESSAGES.extend(plugin_instructions)
    MESSAGES.extend(read_instructions("for_all_outro"))
    LOGGER.debug(MESSAGES)
    rawcontent = send_messages(MESSAGES)


def main(args):
    """
    The main function that is responsible for orchestrating the chat model's operations.

    This function updates the OpenAI API base if a value is provided, initializes the 
    logging level based on user arguments, updates global variables for chat completion 
    arguments based on command-line parameters, sets up instructions based on the model,
    and starts the dialog based on user arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Warnings:
        The function warns the user if  
        an instruction role different than 'user' is used with a Vicuna model.
    """

    global LOG_FORMAT
    global INSTRUCTION_ROLE

    # Update the OpenAI API base if a value is provided
    if args.openai_api_base:
        openai.api_base = args.openai_api_base
        # openai_api_client.set_baseurl(args.openai_api_base)

    if args.log_level.upper() == "SILENT":
        numeric_log_level = logging.CRITICAL + 1
    else:
        numeric_log_level = logging.getLevelName(args.log_level.upper())

    # Update the LOGGER.basicConfig initialization to use the specified logging level
    logging.basicConfig(level=numeric_log_level, format=LOG_FORMAT)

    # Update the global variable chat_completion_args with command-line parameters
    CHAT_COMPLETION_ARGS
    CHAT_COMPLETION_ARGS["model"] = args.model
    CHAT_COMPLETION_ARGS["temperature"] = args.temperature
    CHAT_COMPLETION_ARGS["stream"] = not args.disable_streaming
    CHAT_COMPLETION_ARGS["max_tokens"] = 500

    # if streaming make sure to go to line before logging.

    if not args.disable_streaming:
        LOG_FORMAT = "\n" + LOG_FORMAT

    INSTRUCTION_ROLE = args.instruction_role

    # Don't set streaming to false or api will fail if not supported...

    if "vicuna" in args.model and args.instruction_role != "user":
        warnings.warn(
            "Using a Vicuna model and instruction role different than 'user' is not recommended."
        )

    if args.model_instructions == "model":
        LOGGER.info("Sending instructions to model")
        set_instructions(args.model)
    else:
        LOGGER.info("Sending instructions to model (%s)",args.model)
        set_instructions(args.model_instructions)

    start_dialog(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure the AI model and API settings."
    )
    parser.add_argument(
        "--model", default="gpt-3.5-turbo-0301", help="Specify the model to use."
    )
    parser.add_argument(
        "--model-instructions",
        default="model",
        help="Specify the model instructions to use.",
    )
    parser.add_argument(
        "--disable-plugin-invocation",
        action="store_true",
        default=False,
        help="Disable plugin invocation.",
    )

    parser.add_argument(
        "--disable-streaming",
        action="store_true",
        default=False,
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--instruction-role",
        default="system",
        choices=["system", "user"],
        help="Specify the instruction role.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Specify the temperature."
    )
    parser.add_argument(
        "--disable-spinner", action="store_true", default=False, help="Disable spinner."
    )
    parser.add_argument(
        "--hide-raw-plugin-reponse",
        action="store_true",
        default=False,
        help="Hide plugin raw reponse.",
    )

    parser.add_argument(
        "--print-raw-plugins-output",
        action="store_true",
        default=False,
        help="Print raw plugins response to console.",
    )

    parser.add_argument("--prompt", default="", type=str, help="Send a prompt")
    parser.add_argument(
        "--cli",
        action="store_true",
        default=False,
        help="Enable CLI mode (exit after first answer).",
    )  # New argument
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SILENT"],
        help="Specify the logging level.",
    )
    # Check for the OPENAI_API_BASE environment variable and set the default value
    default_openai_api_base = os.environ.get("OPENAI_API_BASE", None)

    parser.add_argument(
        "--openai_api_base",
        default=default_openai_api_base,
        help="Specify the OpenAI API base URL.",
    )
    parser.add_argument(
        "--openai_api_key",
        default=os.environ.get("OPENAI_API_KEY"),
        required=not os.environ.get("OPENAI_API_KEY"),
        help="Specify the OpenAI API key.",
    )

    cmdline_args = parser.parse_args()
    main(cmdline_args)
