from typing import List, Union, Optional, Type
from gentopia.model.param_model import OpenAIParamModel
from gentopia.agent.base_agent import BaseAgent
from gentopia.llm import OpenAIGPTClient
from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentType, AgentOutput
from gentopia.utils.cost_helpers import calculate_cost
from gentopia.tools import BaseTool
from pydantic import create_model, BaseModel

from gentpool.bench.grader.base import BaseGrader
from gentpool.bench.prompt import *


class DojoGrader(BaseGrader):
    """
    A "dojo" Grader decide if the first prediction win, tie or lose to the second prediction.
    """
    name: str = "DojoGrader"
    type: AgentType = AgentType.vanilla
    grader_type = "dojo"
    version: str = ""
    description: str = "An agent judging which side, left or right, better solves a given task. Input contains a task, a ground truth, left side answer and right side answer. Output either 'left' or 'right' or 'tie'."
    target_tasks: list[str] = []
    llm: BaseLLM = OpenAIGPTClient(model_name="gpt-4", params=OpenAIParamModel(temperature=0))
    prompt_template: PromptTemplate = TeacherStudentDojoPrompt
    plugins: List[Union[BaseTool, BaseAgent]] = []
    examples: Union[str, List[str]] = None

    args_schema: Optional[Type[BaseModel]] = create_model("GateArgsSchema", task=(str, ...),
                                                          ground_truth=(str, ...),
                                                          left=(str, ...),
                                                          right=(str, ...), )

    def run(self, task: str, ground_truth: str, left: str, right: str) -> AgentOutput:
        total_cost = 0
        total_token = 0

        prompt = self.prompt_template.format(task=task, ground_truth=ground_truth, left=left, right=right)

        response = self.llm.completion(prompt)
        if response.state == "error":
            raise ValueError(f"{self.name} fails to retrieve response from LLM.")

        total_cost += calculate_cost(self.llm.model_name, response.prompt_token,
                                     response.completion_token)
        total_token += response.prompt_token + response.completion_token

        return AgentOutput(output=response.content, cost=total_cost, token_usage=total_token)

    def stream(self, *args, **kwargs) -> AgentOutput:
        raise NotImplementedError("GateGrader does not support streaming.")
