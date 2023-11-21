"""Defines the Chained prompt template type."""

from typing import Any, List

from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMessage, PromptValue

from langchain_contrib.utils import f_join, safe_inputs

from .schema import Templatable, into_template
from .z_base import ZStringPromptTemplate


class ChainedPromptValue(PromptValue):
    """A prompt value consisting of smaller prompt values."""

    joiner: str = ""
    """How to join each prompt value together.

    Only used when joining to_string.
    """
    subvalues: List[PromptValue]

    def to_string(self) -> str:
        """Join prompt values together as a single string."""
        return f_join(self.joiner, [x.to_string() for x in self.subvalues])

    def to_messages(self) -> List[BaseMessage]:
        """Append all prompt values together as messages."""
        messages = [
            message for subvalue in self.subvalues for message in subvalue.to_messages()
        ]
        return messages


class ChainedPromptTemplate(ZStringPromptTemplate):
    """A prompt template composed of multiple other prompt templates chained together.

    This is a StringPromptTemplate rather than a BasePromptTemplate to enable use in
    BaseStringMessagePromptTemplate.
    """

    joiner: str = ""
    """How to join each template output together.

    Only meaningful for StringPromptTemplate's.
    """
    subprompts: List[BasePromptTemplate]

    def __init__(self, subprompts: List[Templatable], joiner: str = "", **kwargs: Any):
        """Initialize a ChainedPromptTemplate.

        subprompts can be passed in as just plain strings for convenience.
        """
        prompts = [into_template(p) for p in subprompts]
        kwargs["joiner"] = joiner
        kwargs["subprompts"] = prompts
        kwargs["input_variables"] = list(
            set([var for subprompt in prompts for var in subprompt.input_variables])
        )
        super().__init__(**kwargs)

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs."""
        return self.format_prompt(**kwargs).to_string()

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format each series of prompts with the given inputs."""
        unused_args = set(kwargs.keys())
        values: List[PromptValue] = []
        for subprompt in self.subprompts:
            prompt_args = safe_inputs(subprompt, kwargs)
            unused_args -= prompt_args.keys()
            values.append(subprompt.format_prompt(**prompt_args))
        if unused_args:
            raise KeyError(unused_args)

        return ChainedPromptValue(joiner=self.joiner, subvalues=values)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "chained"
