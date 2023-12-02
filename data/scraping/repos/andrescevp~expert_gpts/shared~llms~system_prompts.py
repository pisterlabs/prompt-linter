import os

import yaml
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessage

from shared.config import DefaultAgentTools, ExpertItem, Prompts

PROMPTS_FILE_PATH = os.getenv("PROMPTS_FILE_PATH", "shared/experts_gpt.yaml")

with open(PROMPTS_FILE_PATH) as f:
    config_prompts = yaml.safe_load(f)


open_ai_prompt_template = """
{
    "role": "{{ role }}",
    "content": "{{ content }}",
    "name": "{{ name }}",
}
"""


def get_open_ai_prompt_template() -> PromptTemplate:
    return PromptTemplate.from_template(
        open_ai_prompt_template, template_format="jinja2"
    )


PROMPT_TOOL_ENGINEER_SYSTEM_PROMPT = SystemMessage(
    **config_prompts["prompt_tool_engineer"]["system"]
)

PROMPT_TOOL_ENGINEER_HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(
    **config_prompts["prompt_tool_engineer"]["user"]
)

PROMPT_ENGINEER_SYSTEM_PROMPT = SystemMessage(
    **config_prompts["prompt_engineer"]["system"]
)

PROMPT_ENGINEER_HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(
    **config_prompts["prompt_engineer"]["user"]
)

GENERAL_EXPERT = ExpertItem(
    **{
        **config_prompts[DefaultAgentTools.GENERALIST_EXPERT.value],
        "prompts": Prompts(
            **config_prompts[DefaultAgentTools.GENERALIST_EXPERT.value]["prompts"]
        ),
    },
)
NO_CODE_FUNCTIONS = ExpertItem(
    **{
        **config_prompts[DefaultAgentTools.NO_CODE_PYTHON_FUNCTIONS_EXPERT.value],
        "prompts": Prompts(
            **config_prompts[DefaultAgentTools.NO_CODE_PYTHON_FUNCTIONS_EXPERT.value][
                "prompts"
            ]
        ),
    },
)
THINKER_EXPERT = ExpertItem(
    **{
        **config_prompts[DefaultAgentTools.THINKER_EXPERT.value],
        "prompts": Prompts(
            **config_prompts[DefaultAgentTools.THINKER_EXPERT.value]["prompts"]
        ),
    },
)
SUMMARIZER_EXPERT = ExpertItem(
    **{
        **config_prompts[DefaultAgentTools.SUMMARIZER_EXPERT.value],
        "prompts": Prompts(
            **config_prompts[DefaultAgentTools.SUMMARIZER_EXPERT.value]["prompts"]
        ),
    },
)
BASE_EXPERTS = {
    DefaultAgentTools.GENERALIST_EXPERT.value: GENERAL_EXPERT,
    DefaultAgentTools.NO_CODE_PYTHON_FUNCTIONS_EXPERT.value: NO_CODE_FUNCTIONS,
    DefaultAgentTools.THINKER_EXPERT.value: THINKER_EXPERT,
    DefaultAgentTools.SUMMARIZER_EXPERT.value: SUMMARIZER_EXPERT,
}
CHAT_HUMAN_PROMPT_TEMPLATE = HumanMessagePromptTemplate.from_template(
    config_prompts["chat_human_prompt_template"],
    role="user",
)

CHAT_SYSTEM_PROMPT_STANDALONE_QUESTION = SystemMessage(
    **config_prompts["standalone_question_prompt_template"]["system"]
)

GET_MEMORIES_TOOL_PROMPT = config_prompts["embedding_tools"][
    "get_embedding_tool_description"
]

SAVE_MEMORIES_TOOL_PROMPT = config_prompts["embedding_tools"][
    "save_embedding_tool_description"
]


PLANNER_SYSTEM_PROMPT = config_prompts["planner_system_prompt"]
