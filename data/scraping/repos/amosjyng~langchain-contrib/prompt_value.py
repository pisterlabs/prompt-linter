"""Module defining prompts involving choices."""

from __future__ import annotations

from typing import List

from fvalues import F
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import BaseMessage, PromptValue
from pydantic import Extra


class ChoiceStr(F):
    """String that keeps track of choices used to create this string."""

    choices: List[str]

    def __new__(cls, f_str: F, choices: List[str]) -> ChoiceStr:
        """Create a new ChoiceStr with cached choices."""
        result = super().__new__(cls, f_str, parts=f_str.parts)
        result.choices = choices
        return result


class BaseChoicePrompt(PromptValue):
    """A prompt that involves picking from a number of choices.

    This is just a wrapper around a regular PromptValue that preserves the choice
    information.
    """

    prompt: PromptValue
    """The encapsulated prompt that provides the actual prompt value."""
    choices: List[str]
    """The list of choices to choose from."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.prompt.to_string()

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return self.prompt.to_messages()

    def _wrap_choice_str(self, str_prompt: str) -> ChoiceStr:
        """Wrap a prompt in ChoiceStr before returning."""
        if isinstance(str_prompt, F):
            f_prompt = str_prompt
        else:
            f_prompt = F(str_prompt)
        return ChoiceStr(f_prompt, self.choices)

    @classmethod
    def from_prompt(cls, prompt: PromptValue, choices: List[str]) -> BaseChoicePrompt:
        """Create one of the child classes from a base prompt."""
        if isinstance(prompt, StringPromptValue):
            return StringChoicePrompt(
                text=prompt.to_string(), prompt=prompt, choices=choices
            )
        elif isinstance(prompt, ChatPromptValue):
            return ChatChoicePrompt(
                messages=prompt.to_messages(), prompt=prompt, choices=choices
            )
        else:
            return cls(prompt=prompt, choices=choices)


class StringChoicePrompt(BaseChoicePrompt, StringPromptValue):
    """A string prompt that involves picking from a number of choices."""

    def to_string(self) -> str:
        """Return prompt as string."""
        str_prompt = self.prompt.to_string()
        return self._wrap_choice_str(str_prompt)

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return self.prompt.to_messages()


class ChatChoicePrompt(BaseChoicePrompt, ChatPromptValue):
    """A chat prompt that involves picking from a number of choices."""

    def to_string(self) -> str:
        """Return prompt as string."""
        str_prompt = self.prompt.to_string()
        return self._wrap_choice_str(str_prompt)

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return self.prompt.to_messages()
