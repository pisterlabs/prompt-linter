import re
import time
import uuid
import os
import logging
from datetime import datetime
from typing import List, Optional, Union, Sequence, Callable, Any, Dict, Tuple

import openai
from langchain.agents import (
    Tool,
    LLMSingleActionAgent,
    AgentOutputParser,
    BaseSingleActionAgent,
    BaseMultiActionAgent,
)
from langchain.agents.agent import ExceptionTool
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from langchain.tools.base import BaseTool
from langchain.agents.tools import InvalidTool

from jarvis.smartgpt import gpt
from jarvis.agent.jarvis_agent import JarvisAgent, EMPTY_FIELD_INDICATOR

# Set up the base react template
react_prompt_template = """Answer the following question as best you can.
Some Performance Evaluations to follow:
- Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
- Constructively self-criticize your big-picture behavior constantly.
- Reflect on past decisions and strategies to refine your approach.
- Every action has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must to answer
Thought: you should always think about what to do
Action: the action to take. Choose the most appropriate action name from the following tool name list: [{tool_names}]. Don't take the useless tools.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final achieve to the original input objective

Begin! Give your final answer and follow the above format.

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class ReactPrompt(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\n"
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class ReactOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL | re.IGNORECASE)
        if not match:
            print(f"Could not parse LLM output: `{llm_output}`")
            return AgentAction(
                tool=llm_output.strip(" ").strip('"'),
                tool_input=llm_output.strip(" ").strip('"'),
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


def setup_react_planner(
    tools: List[Tool], model: str = "gpt-4"
) -> BaseSingleActionAgent:
    llm = gpt.OPEN_AI_MODELS_HUB[model].get_llm()

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(
        llm=llm,
        prompt=ReactPrompt(
            template=react_prompt_template,
            tools=tools,
            input_variables=["input", "intermediate_steps"],
        ),
    )
    return LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=ReactOutputParser(),
        stop=["\nObservation:"],
    )


class AgentExecutor:
    """Consists of an agent using tools."""

    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    tools: Sequence[BaseTool]
    return_intermediate_steps: bool = False
    max_iterations: Optional[int] = 5
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"
    handle_parsing_errors: Union[
        bool, str, Callable[[OutputParserException], str]
    ] = False
    name_to_tool_map: dict[str, BaseTool]

    def __init__(self, tools: List[Tool], model: str = "gpt-4"):
        self.tools = tools
        self.agent = setup_react_planner(tools, model=model)
        self.name_to_tool_map = {tool.name: tool for tool in self.tools}

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
    ) -> Dict[str, Any]:
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _decide_next_step(
        self, inputs: Dict[str, str], intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[AgentAction, AgentFinish]:
        try:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                **inputs,
            )
            return output
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction(ExceptionTool.name, observation, text)
            return output

    def _take_next_step(
        self,
        action: AgentAction,
    ) -> Tuple[AgentAction, str]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        result = []
        # Otherwise we lookup the tool
        if action.tool in self.name_to_tool_map:
            tool = self.name_to_tool_map[action.tool]
            # We then call the tool on the tool input to get an observation
            observation = str(tool.run(action.tool_input))
        else:
            invalid_tool_args = {
                "requested_tool_name": action.tool,
                "available_tool_names": self.name_to_tool_map.keys(),
            }
            observation = InvalidTool().run(invalid_tool_args)
        return (action, observation)

    def _return_stopped_response(
        self,
        question: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        # `force` just returns a constant string

        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\n"

        task_prompt = (
            f"Based on these factual information:\n{thoughts}\n"
            f"\nAnswer the question: {question}. Answer="
        )

        response = gpt.complete(prompt=task_prompt, model=gpt.GPT_4)
        return AgentFinish({"output": response}, task_prompt)

    def run(
        self,
        input: str,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        inputs = {"input": input}

        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step = self._decide_next_step(
                inputs,
                intermediate_steps,
            )
            if isinstance(next_step, AgentFinish):
                return self._return(next_step, intermediate_steps)

            print(f"\nNext action:\n{next_step.log}\n")

            next_step_output = self._take_next_step(next_step)
            intermediate_steps.append(next_step_output)
            print(
                f"\nAction: {next_step_output[0].tool}({next_step_output[0].tool_input})\nOutput:{next_step_output[1]}\n"
            )

            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps)


class JarvisAgentTools:
    def __init__(self, objective: str):
        self.agent = JarvisAgent()
        self.objective = objective
        self.previous_tasks = []
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.subdir = f"{unique_id}-{timestamp}"
        print(f"initial jarvis agent under subdir: {self.subdir}")

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def description(self) -> str:
        return self.agent.description

    def exec(self, tool_input: str) -> str:
        while True:
            task_info = self.agent.execute(
                self.subdir, self.objective, tool_input, self.previous_tasks
            )
            assert task_info is not None, "last_task_info is None"
            if task_info.result != EMPTY_FIELD_INDICATOR:
                break
            print(f"Retring.... cause of empty result of task: {task_info}")

        self.previous_tasks.append(task_info.task_num)
        return task_info.result


if __name__ == "__main__":
    # Logging file name and line number
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    )
    os.makedirs("workspace", exist_ok=True)
    os.chdir("workspace")

    objective = """
    Compose a captivating tweet about the trending AI projects from the last 28 days, using trending data from https://ossinsight.io/collections/artificial-intelligence/.  Here's how to do it:

    Success Criteria:

    - The tweet must summarizes overall trends in AI projects from the last 28 days.
    - 1-3 specific projects need to be featured in the tweet. These projects may rise rapidly in rankings, or github stars count growth rate is ahead of other projects. Make sure your selection is diverse to represent different observed trends.
    - Collect and summarize recent developments (news) of selected projects to ensure that news is timely (nearly a month, current Date: 2023-07-27) and eye-catching
    - The tweet should be engaging, amusing, and adheres to the Twitter's character limit.

    Current Date: 2023-07-27
    """

    jarvis = JarvisAgentTools(objective)
    jarvisTool = Tool(
        name=jarvis.name, description=jarvis.description, func=jarvis.exec
    )
    agent = AgentExecutor([jarvisTool], model=gpt.GPT_4)

    agent.run(objective)
