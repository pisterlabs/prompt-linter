"""Module that defines the choice prompt."""
from __future__ import annotations

from typing import Any, Callable, Generic, List, Sequence, TypeVar, Union

from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import BaseMessage
from pydantic import Field

from langchain_contrib.prompts.z_base import (
    DefaultsTo,
    ZBasePromptTemplate,
    ZChatPromptTemplate,
    ZPromptTemplate,
)
from langchain_contrib.utils import f_join

from .prompt_value import BaseChoicePrompt

ChoicesFormatter = Callable[[List[str]], str]


def get_simple_joiner(joiner: str = ", ") -> ChoicesFormatter:
    """Get a choice formatter that's just a simple joining of strings."""

    def simple_join_choice(choices: List[str]) -> str:
        """Do a simple join on the choice strings."""
        return f_join(joiner, choices)

    return simple_join_choice


def get_oxford_comma_formatter(conjunction: str = "or") -> ChoicesFormatter:
    """Get a choice formatter that respects the Oxford comma."""

    def oxford_comma_list(choices: List[str]) -> str:
        """Phrase the list using the Oxford comma."""
        if len(choices) == 0:
            return ""
        elif len(choices) == 1:
            return choices[0]
        elif len(choices) == 2:
            return f_join(f" {conjunction} ", choices)
        else:
            head = f_join(", ", choices[:-1])
            return f_join(f", {conjunction} ", [head, choices[-1]])

    return oxford_comma_list


def list_of_choices(choices: List[str]) -> str:
    """Return a numerical list of choices."""
    return f_join("\n", [f"{i+1}. {choice}" for i, choice in enumerate(choices)])


T = TypeVar("T")


class ChoicePromptTemplate(ZBasePromptTemplate, Generic[T]):
    """A wrapper prompt template for picking from a number of choices.

    This template preserves choice information in prompts.
    """

    """The base template that this class wraps around."""
    choice_serializer: Callable[[T], str] = lambda x: str(x)
    """How to turn the choices into strings."""
    choices_formatter: ChoicesFormatter = Field(
        default_factory=get_oxford_comma_formatter
    )
    """How to convert from the list of choices to a single string.

    Utility functions to help with this include:

    - get_simple_joiner
    - get_oxford_comma_formatter
    - list_of_choices
    """
    choice_format_key: str = "choices"
    """Which string is used for formatting choices in the template."""

    @classmethod
    def from_base_template(
        cls, base_template: BasePromptTemplate, **kwargs: Any
    ) -> ChoicePromptTemplate:
        """Wrap around a base template."""
        result = super().from_base_template(base_template=base_template, **kwargs)
        assert isinstance(result, ChoicePromptTemplate)
        return result

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> ChoicePromptTemplate:
        """Load a ChoicePromptTemplate from a text template."""
        base_template = ZPromptTemplate.from_template(template)
        return cls.from_base_template(base_template=base_template, **kwargs)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Union[BaseMessagePromptTemplate, BaseMessage]],
        **kwargs: Any,
    ) -> ChoicePromptTemplate:
        """Load a ChoicePromptTemplate from message templates."""
        base_template = ZChatPromptTemplate.from_messages(messages)
        return cls.from_base_template(base_template=base_template, **kwargs)

    def permissive_partial(self, **kwargs: Any) -> ChoicePromptTemplate:
        """Return a partial of the prompt template.

        Permissive version that allows for arbitrary input types.
        """
        if self.choice_format_key in kwargs:
            choices = kwargs[self.choice_format_key]
            assert isinstance(choices, list) or isinstance(choices, DefaultsTo), (
                "Choices must be passed in as list, but is instead "
                f"{type(choices).__name__}: {choices}"
            )

        result = super().permissive_partial(**kwargs)
        assert isinstance(result, ChoicePromptTemplate)
        return result

    @property
    def _prompt_type(self) -> str:
        return "choice"

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs."""
        return self.format_prompt(**kwargs).to_string()

    def format_prompt(self, **kwargs: Any) -> BaseChoicePrompt:
        """Format the prompt while preserving the choices."""
        kwargs = self._merge_partial_and_user_variables(**kwargs)

        if self.choice_format_key not in kwargs:
            raise ValueError(
                f"Choice key '{self.choice_format_key}' not in args: {kwargs}"
            )
        choices = kwargs[self.choice_format_key]
        assert isinstance(choices, list), (
            "Choices must be passed in as list, but is instead "
            f"{type(choices).__name__}: {choices}"
        )
        str_choices = [self.choice_serializer(c) for c in choices]
        kwargs[self.choice_format_key] = self.choices_formatter(str_choices)

        assert (
            self.base_template is not None
        ), "ChoicePromptTemplate requires a base template to be provided"
        prompt = self.base_template.format_prompt(**kwargs)
        return BaseChoicePrompt.from_prompt(prompt, choices=str_choices)
