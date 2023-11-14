"""Module defining a more flexible BasePromptTemplate."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Union

from langchain.prompts.base import (
    BasePromptTemplate,
    StringPromptTemplate,
    StringPromptValue,
    check_valid_template,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import PromptValue
from pydantic import BaseModel, Extra, Field, root_validator


class DefaultsTo(BaseModel):
    """Marks one prompt key as defaulting to another one."""

    default_key: str
    """Default key to get prompt value from."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def __init__(self, default_key: str, **kwargs: Any) -> None:
        """Create a new DefaultTo partial value."""
        super().__init__(default_key=default_key, **kwargs)


class ZBasePromptTemplate(BasePromptTemplate):
    """A prompt template class that allows for arbitrary partials."""

    base_template: Optional[BasePromptTemplate] = None
    """The actual template that this class wraps around.

    If None, then this class is assumed to be overridden.
    """
    permissive_partial_variables: Mapping[str, Any] = Field(default_factory=dict)
    """Partial variables of any type.

    The BasePromptTemplate.format and format_prompt functions take in any arbitrary
    types, so why shouldn't partials as well?
    """

    @classmethod
    def from_base_template(
        cls, base_template: BasePromptTemplate, **kwargs: Any
    ) -> ZBasePromptTemplate:
        """Wrap around a base template."""
        if "input_variables" not in kwargs:
            kwargs["input_variables"] = base_template.input_variables
        return cls(base_template=base_template, **kwargs)

    def format(self, **kwargs: Any) -> str:
        """Format prompt template as a string."""
        return self.format_prompt(**kwargs).to_string()

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with partials taken care of."""
        raise NotImplementedError(
            "Either override _format_prompt or supply a base template"
        )

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt from the base prompt."""
        new_kwargs = self._merge_partial_and_user_variables(**kwargs)
        if self.base_template:
            return self.base_template.format_prompt(**new_kwargs)
        else:
            return self._format_prompt(**new_kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the type of prompt this is."""
        assert (
            self.base_template is not None
        ), "Either override _prompt_type or supply a base template"
        return self.base_template._prompt_type

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> ZBasePromptTemplate:
        """Return a partial of the prompt template."""
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["permissive_partial_variables"] = {
            **self.permissive_partial_variables,
            **kwargs,
        }
        return type(self)(**prompt_dict)

    def permissive_partial(self, **kwargs: Any) -> ZBasePromptTemplate:
        """Return a partial of the prompt template.

        Permissive version that allows for arbitrary input types.
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["permissive_partial_variables"] = {
            **self.permissive_partial_variables,
            **kwargs,
        }
        return type(self)(**prompt_dict)

    def _prep_partials(self, kwargs: Dict[str, Any]) -> Any:
        """Update the kwargs."""

        def eval_partial(value: Any) -> Any:
            if callable(value):
                return value()
            elif isinstance(value, DefaultsTo):
                return kwargs[value.default_key]
            else:
                return value

        return {k: eval_partial(v) for k, v in kwargs.items()}

    def _combined_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all kwargs into one dict."""
        return {
            **self.partial_variables,
            **self.permissive_partial_variables,
            **kwargs,
        }

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        """Merge all partials, including permissive ones."""
        combined_kwargs = self._combined_kwargs(kwargs)
        processed_kwargs = self._prep_partials(combined_kwargs)
        return processed_kwargs


class ZStringPromptTemplate(ZBasePromptTemplate, StringPromptTemplate):
    """A version of StringPromptTemplate with extended flexibility."""


class ZPromptTemplate(ZBasePromptTemplate, PromptTemplate):
    """A version of PromptTemplate with extended flexibility."""

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> ZPromptTemplate:
        """Load a prompt template from a template."""
        result = super().from_template(template, **kwargs)
        assert isinstance(result, ZPromptTemplate)
        return result

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        return StringPromptValue(text=PromptTemplate.format(self, **kwargs))

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        if values["validate_template"]:
            all_inputs = (
                values["input_variables"]
                + list(values["partial_variables"])
                + list(values["permissive_partial_variables"])
            )
            check_valid_template(
                values["template"], values["template_format"], all_inputs
            )
        return values


class ZChatPromptTemplate(ZBasePromptTemplate, ChatPromptTemplate):
    """A version of ChatPromptTemplate with extended flexibility."""

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        return ChatPromptTemplate.format_prompt(self, **kwargs)

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> ZBasePromptTemplate:
        """Return a partial of the chat prompt template."""
        return ZBasePromptTemplate.partial(self, **kwargs)
