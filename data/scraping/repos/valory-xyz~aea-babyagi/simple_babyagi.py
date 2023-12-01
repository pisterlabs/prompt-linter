"""
Heavily inspired by: https://github.com/yoheinakajima/babyagi

python simple_babyagi.py "develop a task list" "solve world hunger"
"""

import os
import sys
import openai
import time
from collections import deque
from dotenv import load_dotenv

# import functions used to build the agent's actions
from actions import (
    task_creation_prompt_builder,
    task_creation_handler,
    task_prioritization_prompt_builder,
    task_prioritization_handler,
    task_execution_prompt_builder,
    task_execution_handler,
    task_stop_or_not_prompt_builder,
    task_stop_or_not_handler,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# flag to stop the procedure
STOP_PROCEDURE = False

# Definition of the action types for the simple agent
action_types = {
    "task_creation": {
        "prompt_builder": task_creation_prompt_builder,
        "handler": task_creation_handler,
    },
    "task_prioritization": {
        "prompt_builder": task_prioritization_prompt_builder,
        "handler": task_prioritization_handler,
    },
    "task_execution": {
        "prompt_builder": task_execution_prompt_builder,
        "handler": task_execution_handler,
    },
    "task_stop_or_not": {
        "prompt_builder": task_stop_or_not_prompt_builder,
        "handler": task_stop_or_not_handler,
    },
}


def executor(globals_: dict, agent_type: str) -> None:
    """
    execute an action using simple agent

    Args:
        globals_ (dict): The globals dictionary
        agent_type (str): The action type to execute
    """
    # load the action type into "agent"
    agent = action_types[agent_type]
    # build the prompt for the corresponding action type
    builder_ = agent["prompt_builder"]
    # create the corresponding prompt for GPT to execute the action
    #  type "agent" and load it into "prompt"
    prompt = builder_(globals_)
    # call GPT with the corresponding "prompt" to execute the action
    # and load the response from the "prompt" into "response"
    response = openai_call(prompt)
    # handle the response from GPT for the corresponding action type "agent"
    handler_ = agent["handler"]
    handler_(response, globals_)


def main(first_task: str, objective: str):
    # initialize the globals dictionary with "objective"
    # this is simple_agent's state variable which is used to keep track of
    # the task list, current task, and the objective so GPT can reason about them.
    globals_ = {
        "objective": objective,
        "task_list": deque([]),
        "current_task": {},
        "result": {"data": ""},
        "keep_going": True,
    }
    # add the first task to the task list
    globals_["task_list"].append({"id": 1, "name": first_task})

    print("\033[89m\033[1m" + "\n=== Simple Loop babyAGI ONLINE ===" + "\033[0m\033[0m")

    # simple agent loop
    while globals_["keep_going"]:
        # execution
        executor(globals_, "task_execution")
        # creation
        executor(globals_, "task_creation")
        # re-prioritization
        executor(globals_, "task_prioritization")
        if STOP_PROCEDURE:
            executor(globals_, "task_stop_or_not")
        time.sleep(1)


def openai_call(
    prompt: str, use_gpt4: bool = False, temperature: float = 0.5, max_tokens: int = 200
):
    if not use_gpt4:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text.strip()
    else:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    _, first_task, objective = sys.argv
    try:
        main(first_task, objective)
    except KeyboardInterrupt:
        print("\033[89m\033[1m" + "\n======== EXIT ========" + "\033[0m\033[0m")
        pass
