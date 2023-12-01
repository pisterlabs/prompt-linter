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


class InstructedGrader(BaseGrader):
    """
    An Instructed Grader that outputs based on explicit eval instructions.
    """
    name: str = "InstructedGrader"
    type: AgentType = AgentType.vanilla
    grader_type = "gate"
    version: str = ""
    description: str = "Grader agent judging the prediction following explicit eval instructions."
    target_tasks: list[str] = []
    llm: BaseLLM = OpenAIGPTClient(model_name="gpt-4", params=OpenAIParamModel(temperature=0))
    prompt_template: PromptTemplate = InstructionFollowingPrompt
    plugins: List[Union[BaseTool, BaseAgent]] = []
    examples: Union[str, List[str]] = None

    args_schema: Optional[Type[BaseModel]] = create_model("GateArgsSchema", eval_instruction=(str, ...),
                                                          agent_message=(str, ...))

    def run(self, eval_instruction: str, agent_message: str) -> AgentOutput:
        total_cost = 0
        total_token = 0

        prompt = self.prompt_template.format(eval_instruction=eval_instruction, agent_message=agent_message)

        response = self.llm.completion(prompt)
        if response.state == "error":
            raise ValueError(f"{self.name} fails to retrieve response from LLM.")

        total_cost += calculate_cost(self.llm.model_name, response.prompt_token,
                                     response.completion_token)
        total_token += response.prompt_token + response.completion_token

        return AgentOutput(output=response.content, cost=total_cost, token_usage=total_token)

    def stream(self, *args, **kwargs) -> AgentOutput:
        raise NotImplementedError("GateGrader does not support streaming.")
