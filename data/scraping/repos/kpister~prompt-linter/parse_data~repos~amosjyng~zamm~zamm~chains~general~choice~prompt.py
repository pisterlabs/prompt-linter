from typing import List

from langchain_contrib.prompts import ChainedPromptTemplate
from langchain_contrib.prompts import (
    ChoicePromptTemplate as ContribChoicePromptTemplate,
)
from langchain_contrib.prompts import Templatable
from langchain_contrib.prompts.choice import list_of_choices


def ChoicePromptTemplate(
    prefix: Templatable, choices: List[str]
) -> ContribChoicePromptTemplate:
    return ContribChoicePromptTemplate.from_base_template(
        base_template=ChainedPromptTemplate(
            subprompts=[
                prefix,
                "{choices}",
                "You decide it's #",
            ],
            joiner="\n\n",
        ),
        choices=choices,
        choices_formatter=list_of_choices,
    )
