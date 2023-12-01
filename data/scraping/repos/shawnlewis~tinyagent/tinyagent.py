# This is a minimal software developer agent based on gpt-4 that really works.
#
# It writes unit tested software. It iterates to solve problems, and asks for user input
# and guidance as needed.
#
# I recommend starting it and then putting your program spec in a spec.txt file in the
# agent's working directory. Then tell it you've put a spec there, and ask it to create
# a plan.txt containing its plan. If you're happy with that, tell it to go ahead.
#
# You can use the --start_dir argument to point it to an existing directory to work in.
# I recommend telling it to read spec.txt and plan.txt to figure out what it should do.
# You can use this capability to stop and restart the agent, and even modify the agent
# (its system instructions, or the code here) in between executions.
#
# WARNING: there are no safety checks here. This agent can run arbitrary commands on
# your system. In practice I have not seen it try to do anything out of its working.
# But I expect you could get it to do something bad if you tried.

import argparse
import datetime
import json
import os
import subprocess
import time

import openai


# Change this how you like. We need the model to break the work into small tasks, because
# of limited context size. Telling it to use TDD is a good way to get it to break the work
# up.
SYSTEM_PROMPT = """
You are an agent that can write software.
- You are the are one of the world's best programmers.
- You should inspect the current directory and possibly the git history to get up to speed.
- You should commit your work as appropriate.
- You are methodical and tend to write small pieces of functionality, confirming them with tests as you go.
- If a task is complete, or you are stuck, please ask for user input by including "NEED_INPUT" in your message.
- You should only use functions you've been provided with.
- Always provide valid json for function call arguments.
- You can definitely execute python code you write by using run_shell_command.
- You are a TDD wizard, you write tests first, and re-execute your tests frequently.
"""


def run_shell_command(command: str) -> tuple[int, str, str]:
    try:
        completed_process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )
        exit_code = completed_process.returncode
        stdout = completed_process.stdout.strip()
        stderr = completed_process.stderr.strip()
    except Exception as e:
        exit_code = -1
        stdout = ""
        stderr = str(e)

    return json.dumps({"exit_code": exit_code, "stdout": stdout, "stderr": stderr})


run_shell_command_spec = {
    "name": "run_shell_command",
    "description": "Run a shell command and capture its exit code, stdout, and stderr.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    },
    "returns": {
        "type": "object",
        "properties": {
            "exit_code": {
                "type": "integer",
                "description": "The exit code returned by the shell command.",
            },
            "stdout": {
                "type": "string",
                "description": "The standard output from the shell command.",
            },
            "stderr": {
                "type": "string",
                "description": "The standard error from the shell command.",
            },
        },
    },
}


def list_files(directory: str) -> list[str]:
    try:
        return json.dumps(os.listdir(directory))
    except Exception as e:
        return json.dumps([str(e)])


#### Spec
list_files_spec = {
    "name": "list_files",
    "description": "List the names of all files in a directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "The path to the directory."}
        },
        "required": ["directory"],
    },
    "returns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of filenames in the directory.",
    },
}


def write_to_file(path: str, content: str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return "File written successfully."
    except Exception as e:
        return str(e)


write_to_file_spec = {
    "name": "write_to_file",
    "description": "Write text to a file at the given path.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The file path."},
            "content": {"type": "string", "description": "The text to write."},
        },
        "required": ["path", "content"],
    },
    "returns": {"type": "string", "description": "Status message."},
}


def read_from_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return str(e)


read_from_file_spec = {
    "name": "read_from_file",
    "description": "Read text from a file at the given path.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "The file path."}},
        "required": ["path"],
    },
    "returns": {"type": "string", "description": "The text read from the file."},
}


agent_functions = [run_shell_command, list_files, write_to_file, read_from_file]
agent_function_specs = [fn.__globals__[f"{fn.__name__}_spec"] for fn in agent_functions]


def openai_chatcompletion(*args, **kwargs):
    # Retrying version of openai.ChatCompletion.create with exponential backoff.
    sleep_time = 5
    errors = 0
    while True:
        try:
            return openai.ChatCompletion.create(*args, **kwargs)
        except openai.error.RateLimitError as e:
            print(f"RATE LIMIT, sleep({sleep_time})")
            time.sleep(sleep_time)
            sleep_time *= 2
        except openai.error.APIError as e:
            errors += 1
            if errors >= 5:
                print("TOO MANY ERRORS, DROPPING TO DEBUGGER")
                breakpoint()
            print(f"API ERROR, sleep({sleep_time})")
            time.sleep(sleep_time)
            sleep_time *= 2


def summarize(summary, messages):
    prompt = f"Please create a summary of the following conversation. This will be used as context for further conversing.\n\nExisting summary: {summary}\n\nNext messages: {json.dumps(messages)}\n\n"
    response = openai_chatcompletion(
        # model="gpt-3.5-turbo-0613",
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["choices"][0]["message"]["content"]


def run_conversation(start_dir):
    # Summary of the earlier part of the conversation, to handle context overflow.
    summary = ""

    # The set of messages in the conversation, truncated in the case of context overflow.
    messages = []

    # starting message
    messages.append(
        {
            "role": "user",
            "content": f"You are starting in a directory (maybe a git repo) that someone has already worked on. Familiarize yourself with the directory, and then ask for instructions.",
        }
    )

    if start_dir is not None:
        dir_name = start_dir
    else:
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"/tmp/agent_{current_timestamp}"
        os.makedirs(dir_name)
    os.chdir(dir_name)

    print("Working dir: ", dir_name)
    print("ENTER TO START")
    input()

    while True:
        try:
            # The set of all messages we send to OpenAI is:
            # - the system prompt
            # - the summary of the earlier part of the conversation
            # - the tail of messages in the conversation
            all_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if summary:
                all_messages.append(
                    {
                        "role": "system",
                        "content": "The following is a summary of the earlier part of the conversation: "
                        + summary,
                    }
                )
            all_messages += messages
            response = openai_chatcompletion(
                # model="gpt-3.5-turbo-0613",
                model="gpt-4",
                messages=all_messages,
                functions=agent_function_specs,
                function_call="auto",  # auto is default, but we'll be explicit
            )
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                # When we exceed context length, compact the conversatoin
                # by using gpt-4 to summarize the existing summary + first half of messages.
                # Then, continue the conversation with the second half of messages.
                messages_len = len(messages)
                first_half = messages[: messages_len // 2]
                second_half = messages[messages_len // 2 :]
                print("LENGTH EXCEEDED, COMPACTING")
                summary = summarize(summary, first_half)
                print("Summary: ", summary)
                messages = second_half
                continue
            else:
                raise e

        response_message = response["choices"][0]["message"]
        messages.append(response_message)

        # The response can include conversation-level content, as well as a functoin
        # call.
        if response_message.get("content"):
            print("Agent: ", json.dumps(response_message.get("content")))
            if not response_message.get("function_call"):
                # If there's no function call, ask for user input, the agent is just
                # stating something. There are many ways you could change this part
                # of the code.
                # if "NEED_INPUT" in response_message["content"]:
                user_input = input("User: ")
                messages.append({"role": "user", "content": user_input})
        if response_message.get("function_call"):
            # The agent is calling a function. Call it!
            function_name = response_message["function_call"]["name"]
            function_to_call = globals()[function_name]
            function_response = None
            try:
                function_args = json.loads(
                    response_message["function_call"]["arguments"]
                )
                print("Agent calling function: ", function_name, end=" ")
                if len(function_args) > 0:
                    print(list(function_args.values())[0], end=" ")
                if len(function_args) > 1:
                    print("...")
                else:
                    print()
            except json.decoder.JSONDecodeError as e:
                # This happens periodically, because the model generates invalid json.
                # We'll write the parse exception as the response for the function call
                # and the model will fix it on the next call! (I have not seen this fail)
                function_response = str(e)
                print("  Function call error:", function_name, "ARGUMENT PARSE ERROR")
            if not function_response:
                function_response = function_to_call(**function_args)
            print("  Function result:", function_response[:100] + "...")

            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )
        print()


def main():
    parser = argparse.ArgumentParser(description="Agent")
    parser.add_argument("--start_dir", type=str, help="Starting directory")

    args = parser.parse_args()
    start_dir = args.start_dir

    run_conversation(start_dir)


if __name__ == "__main__":
    main()
