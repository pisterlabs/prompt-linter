import time

from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM

from geospatial_agent.agent.geospatial.planner.prompts import _graph_generation_instructions, \
    _graph_reply_example, _task_name_generation_prompt, _graph_requirement_list, \
    _planning_graph_task_prompt_template
from geospatial_agent.shared.prompts import GIS_AGENT_ROLE_INTRO, HUMAN_STOP_SEQUENCE
from geospatial_agent.shared.utils import extract_code


class PlannerException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def gen_task_name(llm: LLM, task: str) -> str:
    """Returns a task name for creating unix folders from task description using LLM"""
    task_name_gen_prompt_template: PromptTemplate = PromptTemplate.from_template(_task_name_generation_prompt)
    task_name_gen_prompt = task_name_gen_prompt_template.format(human_role="Human",
                                                                assistant_role="Assistant",
                                                                task_definition=task)
    task_name = llm.predict(text=task_name_gen_prompt, stop=[HUMAN_STOP_SEQUENCE]).strip()
    task_name = f'{int(time.time())}_{task_name}'
    return task_name


def gen_plan_graph(llm: LLM, task_definition: str, data_locations_instructions: str) -> str:
    """Returns a plan graph in the form of python code from a task definition."""
    try:
        graph_plan_code = _gen_plan_graph_code(llm, task_definition, data_locations_instructions)
        return graph_plan_code
    except Exception as e:
        raise PlannerException(f"Failed to generate graph plan code for task") from e


def _gen_plan_graph_code(llm: LLM, task_definition: str, data_locations_instructions: str):
    # Generating a graph plan python code using the LLM.
    graph_requirements = _get_graph_requirements()
    graph_gen_prompt_template: PromptTemplate = PromptTemplate.from_template(_planning_graph_task_prompt_template)
    chain = LLMChain(llm=llm, prompt=graph_gen_prompt_template)
    graph_plan_response = chain.run(human_role="Human",
                                    planner_role_intro=GIS_AGENT_ROLE_INTRO,
                                    graph_generation_instructions=_graph_generation_instructions,
                                    task_definition=task_definition.strip("\n").strip(),
                                    graph_requirements=graph_requirements,
                                    graph_reply_example=_graph_reply_example,
                                    data_locations_instructions=data_locations_instructions,
                                    assistant_role="Assistant",
                                    stop=[HUMAN_STOP_SEQUENCE])
    # Use the LLM to generate a plan graph code
    graph_plan_code = extract_code(graph_plan_response)
    return graph_plan_code


def _get_graph_requirements() -> str:
    """Returns planning graph requirements list"""
    requirements = _graph_requirement_list.copy()
    graph_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(requirements)])
    return graph_requirement_str
