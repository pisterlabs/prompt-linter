import yaml
import openai
import json
import streamlit as st
from secret import keys
from src.gpt_function import GPTFunction, gpt_agent
from functions.gmaps import get_travel_distance
from agents.basic import run_on_list
from data.manipulation import analyze_data, transform_data, undo_transformation
from data.storage import get_data_details


###################################
#            PROMPTS              #
###################################

def _starter_prompt(task: str) -> str:
    return f"""You are an intelligent agent that will be completing a task step by step.
You are efficient, smart, and capable of completing the task. Always taking the most straightforward approach.
First you must break down the task into smaller steps. Format them as a json list of steps.
E.g. {{"steps" : ["Step 1: Do this", "Step 2: Do that", "Step 3: Do the other thing"]}}
Take note of the functions available to you. Use to help complete the task, and take them into account when planning.
Do not break down the task into too many steps, as this will make it harder to complete.
The user does not see this conversation, therefore you cannot ask them any questions.
Do NOT provide any commentary at all whatsoever or information that is not relevant to the task.
Make sure that the output includes all of the information relevant to the task.
Your next response must be a single json plan.
Do not call functions unless it is actually necessary.
Use the most relevant functions.
Always explore the data before using it. Only use columns that actually exist.
Always analyze and get the details of data. Do not use data without knowing what it is.
Always use functions to manipulate the data. Do not simply give code.
The task is: {task}
Do not add any unnecessary steps to the plan."""


def _invalid_plan_prompt() -> str:
    return """The json you provided is not valid. 
Please try again. 
The next message must be a single json plan, do not apologize"""


def _plan_satisfaction_prompt() -> str:
    return """Is this plan sufficient to complete the task? Is it as simple as possible?
Does it contain no unnecessary steps?
If you are happy with the plan respond with 'yes'. If not, respond with 'no'."""


def _replan_prompt() -> str:
    return """Please provide a new plan. The next message must be a single json plan, do not apologize"""


def _step_function_prompt(step: str) -> str:
    return f"""Okay, lets go onto the next step: {step}. 
First, do you need any functions to complete this step?
Remember you may already have all of the information needed above.
If you need a function, respond with 'yes'. If not, respond with 'no'.
If you are unsure, response with 'yes'"""


def _step_prompt(step: str) -> str:
    return f"""Okay, complete step {step}. Do whatever is necessary to complete this step."""


def _step_satisfaction_prompt(step: str) -> str:
    return f"""Has this achieved the goal of step {step}? 
If so, respond with 'yes'. If not, respond with 'no'.
Do no include anything else in the response. Just a "yes" or "no", do not repeat the plan"""


def _step_retry_prompt(step: str) -> str:
    return f"""Please try again to complete step {step}. 
Fix whatever mistake was made. Remember, the user cannot help you"""


def _step_failure_prompt(step: str) -> str:
    return f"""You have failed to complete step {step} too many times. 
Please explain the reason for this failure."""


def _plan_update_question_prompt(steps: list[str]) -> str:
    return f"""The current plan is {steps}. Based on all of the above, does it need to be amended? 
If so, respond with 'yes'. If not, respond with 'no'
Do not include anything else in the response. Just a "yes" or "no", do not repeat the plan"""


def _plan_update_prompt() -> str:
    return """Please amend the plan to include the new step. 
The next message must be a single json plan"""


def _final_summarization_prompt(task: str) -> str:
    return f"""The plan has been completed. 
Based on everything done above, what is the final output for the task of {task}?
The response must contain all useful information uncovered during the completion of the task
Make sure the response is well structured."""


###################################
#              AGENT              #
###################################


class PlanningAgent:
    def __init__(self, functions: list[GPTFunction]):
        openai.api_key = keys.openai_key
        config = yaml.safe_load(open("config.yaml", "r"))
        self.model_name = config["model"]["agent"]
        self.messages = []
        self.functions = {}
        self.max_retries = 3
        for function in functions:
            self.functions[function.name] = function

    def get_response(self, prompt: str, allow_function_calls: bool = True):
        print("\nSystem:")
        print(prompt)
        self.messages.append({"role": "system", "content": prompt})
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.messages,
            functions=list(map(lambda x: x.to_dict(), self.functions.values())),
            function_call="auto" if allow_function_calls else "none"
        )["choices"][0]["message"]
        self.messages.append(response)

        if response.get("function_call") and allow_function_calls:
            func_name = response["function_call"]["name"]
            func_args = response["function_call"]["arguments"]
            func_args = json.loads(func_args)
            self.call_function(func_name, func_args)
            return None
        else:
            print("\nAgent:")
            print(response["content"])

        return response["content"]

    def call_function(self, func_name: str, func_args: dict):
        print("\nFunction call:\n", func_name, "\n", func_args)
        func = self.functions[func_name]
        func_results = func(func_args)
        print("\nFunction results:\n", func_results)
        self.messages.append({"role": "function", "name": func_name, "content": func_results})

    def run(self, task: str):
        prompt = _starter_prompt(task)
        with st.spinner("Planning..."):

            valid_plan = False
            while not valid_plan:
                response = self.get_response(prompt, allow_function_calls=False)
                try:
                    steps = json.loads(response)["steps"]
                except:
                    prompt = _invalid_plan_prompt()
                    continue
                prompt = _plan_satisfaction_prompt()
                response = self.get_response(prompt, allow_function_calls=False)
                lowered = response.lower()
                if "yes" in lowered and lowered.index("yes") == 0:
                    valid_plan = True
                else:
                    prompt = _replan_prompt()

        for i in range(len(steps)):
            step = steps[i]
            completed = False
            retries = 0
            with st.spinner(step):
                step_prompt = _step_function_prompt(step)
                response = self.get_response(step_prompt, allow_function_calls=False)
                allow_function_calls = False
                lowered = response.lower()
                if "yes" in lowered and lowered.index("yes") == 0:
                    allow_function_calls = True
                step_prompt = _step_prompt(step)

                while not completed:
                    self.get_response(step_prompt, allow_function_calls)
                    prompt = _step_satisfaction_prompt(step)
                    response = self.get_response(prompt, allow_function_calls=False)
                    lowered = response.lower()
                    if "yes" in lowered and lowered.index("yes") == 0:
                        completed = True
                    else:
                        retries += 1
                        # If the agent fails to complete the step too many times, it must explain why
                        if retries >= self.max_retries:
                            prompt = _step_failure_prompt(step)
                            response = self.get_response(prompt, allow_function_calls=False)
                            response = "The task could not be completed. Because: " + response
                            response += "\n Inform the user of this and do not try again."
                            return response

                        step_prompt = _step_retry_prompt(step)

                prompt = _plan_update_question_prompt(steps[i:])
                response = self.get_response(prompt, allow_function_calls=False)
                lowered = response.lower()
                if "yes" in lowered and lowered.index("yes") == 0:
                    prompt = _plan_update_prompt()
                    response = self.get_response(prompt, allow_function_calls=False)
                    steps = json.loads(response)["steps"]

        with st.spinner("Finalizing response..."):
            prompt = _final_summarization_prompt(task)
            response = self.get_response(prompt)
        return response


@gpt_agent
def complete_task(task: str):
    """
    Useful for completing complex, multi-step tasks.
    Use this if a task be completed in a single function call.
    :param task: the task you wish to be completed
    """
    print("Task:", task)
    conversator = st.session_state["conversator"]
    # get a copy of functions in the conversator and remove this function from it. It is a dict
    functions = conversator.functions.copy()
    functions.pop("complete_task")
    agent = PlanningAgent(functions.values())
    return {"result": agent.run(task)}


if __name__ == '__main__':
    functions = [
        get_travel_distance,
        analyze_data, transform_data, undo_transformation, get_data_details,
        run_on_list
    ]
    agent = PlanningAgent(functions)
    agent.run("Calculate how long it would take to travel from London to Madrid. While stopping in Paris.")
