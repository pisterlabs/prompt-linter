"""Types useful for prompting."""

from typing import Union

from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import BaseMessage

from .z_base import ZBasePromptTemplate, ZChatPromptTemplate, ZPromptTemplate

Templatable = Union[str, BaseMessagePromptTemplate, BaseMessage, BasePromptTemplate]
"""Anything that can be converted directly into a BasePromptTemplate."""


def into_template(templatable: Templatable) -> BasePromptTemplate:
    """Convert a Templatable into a proper BasePromptTemplate."""
    if isinstance(templatable, str):
        return ZPromptTemplate.from_template(templatable)
    elif isinstance(templatable, BaseMessagePromptTemplate) or isinstance(
        templatable, BaseMessage
    ):
        return ZChatPromptTemplate.from_messages([templatable])
    elif isinstance(templatable, ZBasePromptTemplate):
        return templatable
    elif isinstance(templatable, BasePromptTemplate):
        return ZBasePromptTemplate.from_base_template(templatable)
    else:
        raise ValueError(f"Don't know how to convert {templatable} into template")
