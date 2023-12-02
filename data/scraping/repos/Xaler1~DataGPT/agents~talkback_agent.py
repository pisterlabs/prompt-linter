import openai
import json
import streamlit as st
from src.gpt_function import GPTFunction, gpt_agent
from base_agent import BaseAgent


###################################
#            PROMPTS              #
###################################

def _starter_prompt(task: str) -> str:
    return f"""You are an intelligent agent that will be completing a task step by step.
You are efficient, smart, and capable of completing the task. Always taking the most straightforward approach.
I will guide you through the process, so do not get ahead of yourself.
The user does not see this conversation, therefore you cannot ask them any questions.
Do NOT provide any commentary at all whatsoever or information that is not relevant to the task.
Make sure that the output includes all of the information relevant to the task.
Do not call functions unless it is actually necessary.
Use the most relevant functions.
Always explore the data before using it. Only use columns that actually exist.
Always analyze and get the details of data. Do not use data without knowing what it is.
Always use functions to manipulate the data. Do not simply give code.
Remember that you can extract information from the data using analyze_data and transform_data using transform_data.
The task is: {task}
What should be the first step to completing it?
Your next message must be just a single step and nothing else.
"""


def _step_function_prompt(step: str) -> str:
    return f"""Okay, lets work on this step: "{step}". 
First, do you need any functions to complete this step?
Remember you may already have all of the information needed above.
If you need a function, respond with 'yes'. If not, respond with 'no'.
If you are unsure, response with 'yes'"""


def _step_prompt(step: str) -> str:
    return f"""Okay, complete step "{step}". Do whatever is necessary to complete this step."""


def _step_satisfaction_prompt(step: str) -> str:
    return f"""Has this achieved the goal of step "{step}"? 
If so, respond with 'yes'. If not, respond with 'no'.
Do no include anything else in the response. Just a "yes" or "no", do not repeat the plan"""


def _step_retry_prompt(step: str) -> str:
    return f"""Please try again to complete step "{step}". 
Fix whatever mistake was made.
Fix any code issues there were.
Fix the data if needed in the most sensible way possible.
Make sure to execute the actual code using one of the data processing functions.
Consider undoing changes you did in previous steps. 
Remember, the user cannot help you"""


def _step_failure_prompt(step: str) -> str:
    return f"""You have failed to complete step "{step}" too many times. 
Please explain the reason for this failure."""

def _task_complete_prompt(task: str) -> str:
    return f"""Given all of the above, has the task of "{task}" been completed?
Make sure that you now have all of the information and can provide a final output to the user. 
If so, respond with 'yes'. If not, respond with 'no'.
Do not provide any other output, just a 'yes' or 'no'."""

def _next_step_prompt(task: str) -> str:
    return f"""Now, given all of the above and the task of "{task}".
What should be the next step towards completing this task?"""


def _final_summarization_prompt(task: str) -> str:
    return f"""The plan has been completed. 
Based on everything done above, what is the final output for the task of {task}?
The response must contain all useful information uncovered during the completion of the task
Make sure the response is well structured."""


###################################
#              AGENT              #
###################################


class TalkbackAgent(BaseAgent):
    def __init__(self, functions: list[GPTFunction]):
        super().__init__(functions)

    def run(self, task: str):
        prompt = _starter_prompt(task)
        task_done = False

        with st.spinner("Initializing agent..."):
            step = self.get_response(prompt, allow_function_calls=False)
        while not task_done:
            step_success = False
            retries = 0
            with st.spinner(step):
                #step_prompt = _step_function_prompt(step)
                #response = self.get_response(step_prompt, allow_function_calls=False)
                allow_function_calls = True
                #lowered = response.lower()
                #if "yes" in lowered and lowered.index("yes") == 0:
                #    allow_function_calls = True
                step_prompt = _step_prompt(step)

                while not step_success:
                    self.get_response(step_prompt, allow_function_calls)
                    prompt = _step_satisfaction_prompt(step)
                    response = self.get_response(prompt, allow_function_calls=False)
                    lowered = response.lower()
                    if "yes" in lowered and lowered.index("yes") == 0:
                        step_success = True
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

                prompt = _task_complete_prompt(task)
                response = self.get_response(prompt, allow_function_calls=False)
                lowered = response.lower()
                if "yes" in lowered and lowered.index("yes") == 0:
                    task_done = True
                    break

                prompt = _next_step_prompt(task)
                response = self.get_response(prompt, allow_function_calls=False)
                step = response


        with st.spinner("Finalizing response..."):
            prompt = _final_summarization_prompt(task)
            response = self.get_response(prompt, allow_function_calls=False)
        return response


@gpt_agent
def complete_task(task: str):
    """
    Useful for completing complex, multi-step tasks.
    Use this if a task be completed in a single function call.
    When calling this, DO NOT CALL ANY OTHER FUNCTIONS.
    :param task: the task you wish to be completed
    """
    print("Task:", task)
    conversator = st.session_state["conversator"]
    # get a copy of functions in the conversator and remove this function from it. It is a dict
    functions = conversator.functions.copy()
    functions.pop("complete_task")
    agent = TalkbackAgent(functions.values())
    return {"result": agent.run(task)}
