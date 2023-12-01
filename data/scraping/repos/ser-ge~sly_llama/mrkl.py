"""
Using sly and sly_llama to implement MRKL
"""

from typing import Callable, ContextManager, List, Optional
from pydantic import BaseModel, root_validator

from sly_llama import llm_call, LlmException
from sly_llama.models import LlmOutput

from langchain import OpenAI

from sly_llama import RetryException

# TODO set model name in .env
# TODO factor out langchain lllm, call OpenAI directly
llm = OpenAI(model_name="gpt-4")


class MrklOutput(LlmOutput):
    """
    Model to validate the output of the Mrkl llm call

    Attributes:
        action: The action taken by the Mrkl llm
        action_input: The input to the action
        thought: The thought process of the Mrkl llm
        final_answer: The final answer of the Mrkl llm
    """

    action: Optional[str] = None
    action_input: Optional[str] = None
    thought: Optional[str] = None
    final_answer: Optional[str] = None

    @staticmethod
    @_(r"Thought:((.|\n)*?)(?=Action|Final)")
    def THOUGHT(matches: List):
        return matches[0][1].strip()

    @_(r"Action Input:((.|\n)*?)(?=Observation)")
    def ACTION_INPUT(matches: List):
        return matches[0][1].strip()

    @_(r"Action:((.|\n)*?)(?=Action Input)")
    def ACTION(matches):
        return matches[0][1].strip()

    @_(r"Final Answer:((.|\n |\s)*)")
    def FINAL_ANSWER(matches):
        return matches[0][1].strip()

    @root_validator(pre=True)
    def check_action_or_answer(cls, values):
        """
        Ensure that either an action or a final answer is given.
        """
        action = "action_input" in values and "action" in values
        answer = "final_answer" in values

        if not any([action, answer]):
            raise LlmException(
                "You must either choose an action or give the final answer"
            )

        return values


@llm_call(
    llm,
    stop_sequence="Observation",
    verbose=False,
    return_prompt=True,
    return_llm_output=True,
)
def mrkl_start(tools, tool_names, request) -> MrklOutput:
    """
    You are a helpful assistant designed to answer quesitons.

    You have access to the following tools:

    {tools}

    Use the following format:

    Request: the question you must answer if you can
    Thought: you should always think about what to do
    Action: name of the tool to use, should be one of [{tool_names}]
    Action Input: the input to the tool
    Observation: the result of the tool
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: [
        "is_possible": <boolean indicating if the request has been successfully answered>,
        "explanation": <a description of why and how the request can or cannot be answered>,
        "answer" : <the answer directly addressing the request>
    ]

    Begin!

    Request: {request}

    """


@llm_call(
    llm,
    stop_sequence="\nObservation",
    verbose=False,
    return_prompt=True,
    return_llm_output=True,
)
def mrkl_step(history:str, current_observation:str) -> MrklOutput:
    """
    {history}
    {current_observation}
    """



def insert_newline_after_match(string: str, pattern: str = "Action Input:"):
    """
    Inserts a newline after the given pattern in the given string.
    """
    return string.replace(pattern, pattern + "\n")

def mrkl_agent(
    query: str, tools_list: List, max_iters: int, max_retries: int
) -> str | None:
    """
    Runs the MRKL agent with the given query, tools list, and maximum number of iterations.

    Parameters
    ----------
    query : str
        The query to be used by the MRLKL agent.
    tools_list : List[Tool]
        A list of tools to be used by the MRLKL agent.
    max_iters : int
        The maximum number of iterations to run the MRLKL agent for.
    max_retries : int
        The maximum number of retries to attempt before giving up if LlmException is thrown
    """

    tools = {tool.name: tool for tool in tools_list}
    tool_info = str(({t.name: t.description for t in tools.values()}))
    tool_names = str(tools.keys())

    # Start the MRLKL agent with the initial conditions

    for _ in range(max_retries):

        try:
            mrkl_output, first_prompt, raw_output = mrkl_start(tool_info, tool_names, query)
            break

        except LlmException as e:
            query = query + '\n' + e.message
            error_message = e
    else:
        raise RetryException(error_message)

    last_output = insert_newline_after_match(raw_output, "Action Input:")
    history = first_prompt + last_output

    print(history)

    for _ in range(max_iters):

        # if chosen action in tool run the tool and set observation
        if mrkl_output.action in tools:
            current_observation = tools[mrkl_output.action](mrkl_output.action_input)
        else:
            current_observation = (
                f"{mrkl_output.action} not a valid tool, try another one"
            )

        # run a single mrkl step until the output can be parsed correctly or max_retries is reached
        for _ in range(max_retries):
            try:
                mrkl_output, last_prompt, raw_output = mrkl_step(
                    history, current_observation
                )
                break

            # add error message to observation for the next retry loop
            except LlmException as e:
                current_observation = current_observation + e.message
                error_message =e

        else:
            raise RetryException(f"mrkl_step exceeeded retries, last error: {error_message}")

        # the llm one shot learns better if it can see last action separated by new line, esp code indent
        last_output = insert_newline_after_match(raw_output, "Action Input:")

        history = last_prompt + last_output
        print(last_prompt)
        print(last_output)

        if mrkl_output.final_answer:
            return mrkl_output.final_answer
