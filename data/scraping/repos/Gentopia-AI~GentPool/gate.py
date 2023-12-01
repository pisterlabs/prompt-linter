from typing import List, Union, Optional, Type
from gentopia.model.param_model import OpenAIParamModel
from gentopia.agent.base_agent import BaseAgent
from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentType, AgentOutput
from gentopia.utils.cost_helpers import calculate_cost
from gentopia.tools import BaseTool
from gentopia.llm.client import OpenAIGPTClient
from pydantic import create_model, BaseModel

from gentpool.bench.grader.base import BaseGrader
from gentpool.bench.prompt import *


class GateGrader(BaseGrader):
    """
    A "gate" Grader decide if a given task is "passed" or "failed".
    """
    name: str = "GateGrader"
    type: AgentType = AgentType.vanilla
    grader_type = "gate"
    version: str = ""
    description: str = "Grader agent judging if the prediction to a given task is passed or failed. Input contains a task, a ground truth and a prediction. Output either 'passed' or 'failed'."
    target_tasks: list[str] = []
    llm: BaseLLM = OpenAIGPTClient(model_name="gpt-4", params=OpenAIParamModel(temperature=0))
    prompt_template: PromptTemplate = TeacherStudentGatePrompt
    plugins: List[Union[BaseTool, BaseAgent]] = []
    examples: Union[str, List[str]] = None

    args_schema: Optional[Type[BaseModel]] = create_model("GateArgsSchema", task=(str, ...),
                                                          ground_truth=(str, ...),
                                                          prediction=(str, ...))

    def run(self, task: str, ground_truth: str, prediciton: str) -> AgentOutput:
        total_cost = 0
        total_token = 0

        prompt = self.prompt_template.format(task=task, ground_truth=ground_truth, prediction=prediciton)

        response = self.llm.completion(prompt)
        if response.state == "error":
            raise ValueError(f"{self.name} fails to retrieve response from LLM.")

        total_cost += calculate_cost(self.llm.model_name, response.prompt_token,
                                     response.completion_token)
        total_token += response.prompt_token + response.completion_token

        return AgentOutput(output=response.content, cost=total_cost, token_usage=total_token)

    def stream(self, *args, **kwargs) -> AgentOutput:
        raise NotImplementedError("GateGrader does not support streaming.")


class BatchGateGrader(BaseGrader):
    """
    A "gate" Grader decide if a given task is "passed" or "failed".
    Input should be a list of tasks, ground truths and predictions.
    """
    name: str = "BatchGateGrader"
    type: AgentType = AgentType.vanilla
    grader_type = "gate"
    version: str = ""
    description: str = "Grader agent judging if predictions to given tasks are passed or failed. Input contains a list of tasks, ground truth and predictions. Output a list of 'passed' or 'failed'."
    target_tasks: list[str] = []
    llm: BaseLLM
    prompt_template: PromptTemplate = BatchTeacherStudentGatePrompt
    plugins: List[Union[BaseTool, BaseAgent]] = []
    examples: Union[str, List[str]] = None

    args_schema: Optional[Type[BaseModel]] = create_model("GateArgsSchema", tasks=(List[str], ...),
                                                          ground_truth=(List[str], ...),
                                                          predictions=(List[str], ...))

    def run(self, tasks: List[str], ground_truth: List[str], predicitons: List[str]) -> AgentOutput:
        total_cost = 0
        total_token = 0

        task_chunk, gt_chunk, pred_chunk = self._preprocess(tasks, ground_truth, predicitons)
        prompt = self.prompt_template.format(task=task_chunk, ground_truth=gt_chunk, prediction=pred_chunk)

        response = self.llm.completion(prompt)
        if response.state == "error":
            raise ValueError(f"{self.name} fails to retrieve response from LLM.")

        total_cost += calculate_cost(self.llm.model_name, response.prompt_token,
                                     response.completion_token)
        total_token += response.prompt_token + response.completion_token

        return AgentOutput(output=self._postprocess(response.content, tasks), cost=total_cost, token_usage=total_token)

    def stream(self, *args, **kwargs) -> AgentOutput:
        raise NotImplementedError("GateGrader does not support streaming.")

    def _preprocess(self, tasks, ground_truth, predictions):
        """
        Preprocess the input into a chunk of text.
        For example, task chunk should be:
        (1). task 1
        (2). task 2
        (3). task 3
        """
        if not (len(tasks) == len(ground_truth) == len(predictions)):
            raise ValueError("The number of tasks, ground truths and predictions should be the same.")
        counter = 1
        task_chunk, gt_chunk, pred_chunk = "", "", ""
        for task, gt, pred in zip(tasks, ground_truth, predictions):
            task_chunk += f"({counter}). {task}\n"
            gt_chunk += f"({counter}). {gt}\n"
            pred_chunk += f"({counter}). {pred}\n"
            counter += 1

        return task_chunk, gt_chunk, pred_chunk

    def _postprocess(self, content, tasks):
        """
        Postprocess the output content from LLM.
        """
        parsed = [p for p in content.split("\n") if p]
        parsed = [t.split(").", 1)[1].strip() for t in parsed]
        if len(parsed) != len(tasks):
            raise ValueError(
                "Parsing validation failed for grader output, because the number of grades does not match the number of tasks.")
        return ",".join(parsed)
