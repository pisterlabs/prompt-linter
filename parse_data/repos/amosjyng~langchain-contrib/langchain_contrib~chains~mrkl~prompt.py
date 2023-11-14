"""Prompting configuration for MRKL agents."""
from __future__ import annotations

from typing import Any, Sequence, Union

from fvalues import F
from langchain.base_language import BaseLanguageModel
from langchain.chains.prompt_selector import BasePromptSelector, is_chat_model
from langchain.prompts.base import BasePromptTemplate, StringPromptValue
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseMessage, PromptValue
from pydantic import Extra

from langchain_contrib.prompts import (
    ChainedPromptValue,
    ChoicePromptTemplate,
    DefaultsTo,
    ZBasePromptTemplate,
    ZChatPromptTemplate,
    ZPromptTemplate,
)
from langchain_contrib.prompts.choice import get_simple_joiner


class MrklPromptTemplate(ZBasePromptTemplate):
    """A prompt template that optionally appends the agent scratchpad.

    If {agent_scratchpad} is not found inside the template, it will be appended instead.
    This allows for all of the following:
    - putting the scratchpad as a regular string in a string template
    - putting the scratchpad as a regular string in a message in a chat template
    - putting the scratchpad as a chat in a chat template
    """

    base_template: BasePromptTemplate
    """Base template used for formatting.

    May or may not contain the agent scratchpad key. If it doesn't, then the agent
    scratchpad will be appended to the end.
    """
    scratchpad_key: str = "agent_scratchpad"
    """Which key will be used for agent scratchpad formatting."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_base_template(
        cls,
        base_template: BasePromptTemplate,
        scratchpad_key: str = "agent_scratchpad",
        **kwargs: Any,
    ) -> MrklPromptTemplate:
        """Load a MRKL prompt template from a base template.

        Input variables must at least include {tools}, {tool_descriptions}, and
        {input}. {agent_scratchpad} is optional in the template declaration, but must
        still be passed in at prompt formatting time.
        """
        tool_names_template = ChoicePromptTemplate.from_base_template(
            base_template=base_template,
            choice_format_key="tools",
            choice_serializer=lambda tool: tool.name,
            choices_formatter=get_simple_joiner(),
        )
        tool_descriptions_template = ChoicePromptTemplate.from_base_template(
            base_template=tool_names_template,
            choice_format_key="tool_descriptions",
            choice_serializer=lambda tool: F(f"{tool.name}: {tool.description}"),
            choices_formatter=get_simple_joiner("\n"),
        )
        input_variables = base_template.input_variables.copy()

        if scratchpad_key not in input_variables:
            input_variables.append(scratchpad_key)

        final_partial = (
            super()
            .from_base_template(
                tool_descriptions_template, input_variables=input_variables
            )
            .permissive_partial(tool_descriptions=DefaultsTo("tools"), **kwargs)
        )
        assert isinstance(final_partial, MrklPromptTemplate)
        return final_partial

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> MrklPromptTemplate:
        """Load a ChoicePromptTemplate from a text template."""
        base_template = ZPromptTemplate.from_template(template)
        return cls.from_base_template(base_template=base_template, **kwargs)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Union[BaseMessagePromptTemplate, BaseMessage]],
        **kwargs: Any,
    ) -> MrklPromptTemplate:
        """Load a ChoicePromptTemplate from message templates."""
        base_template = ZChatPromptTemplate.from_messages(messages)
        return cls.from_base_template(base_template=base_template, **kwargs)

    def format(self, **kwargs: Any) -> str:
        """Format the prompt as a string."""
        return self.format_prompt(**kwargs).to_string()

    def _scratchpad_as_str(self, agent_scratchpad: Any) -> str:
        """Ensure that the agent scratchpad becomes a string."""
        if isinstance(agent_scratchpad, str):
            return agent_scratchpad
        elif isinstance(agent_scratchpad, PromptValue):
            return agent_scratchpad.to_string()
        else:
            raise ValueError(f"Unknown agent scratchpad type for {agent_scratchpad}")

    def _scratchpad_as_prompt_value(self, agent_scratchpad: Any) -> PromptValue:
        """Ensure that the agent scratchpad becomes a PromptValue."""
        if isinstance(agent_scratchpad, str):
            return StringPromptValue(text=agent_scratchpad)
        elif isinstance(agent_scratchpad, PromptValue):
            return agent_scratchpad
        else:
            raise ValueError(f"Unknown agent scratchpad type for {agent_scratchpad}")

    @property
    def _prompt_type(self) -> str:
        return "mrkl"

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with the given inputs."""
        assert (
            self.scratchpad_key in kwargs
        ), "Agent scratchpad must still be provided as input key"
        scratchpad = kwargs.pop(self.scratchpad_key)
        if self.scratchpad_key in self.base_template.input_variables:
            kwargs[self.scratchpad_key] = self._scratchpad_as_str(scratchpad)
            return self.base_template.format_prompt(**kwargs)
        else:
            return ChainedPromptValue(
                joiner="\n\n",
                subvalues=[
                    self.base_template.format_prompt(**kwargs),
                    self._scratchpad_as_prompt_value(scratchpad),
                ],
            )


DEFAULT_MRKL_STRING_TEMPLATE = """
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""".lstrip()


def get_string_mrkl_prompt(
    template: str = DEFAULT_MRKL_STRING_TEMPLATE,
) -> MrklPromptTemplate:
    """Get a string version of the MRKL prompt."""
    return MrklPromptTemplate.from_template(template)


DEFAULT_MRKL_SYSTEM_TEMPLATE = """
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tools}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Here is an example of an invalid $JSON_BLOB:

```
{{{{
  "action": $FIRST_TOOL_NAME,
  "action_input": $FIRST_INPUT
}}}}

{{{{
  "action": $SECOND_TOOL_NAME,
  "action_input": $SECOND_INPUT
}}}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.
""".strip()  # noqa


def get_chat_mrkl_prompt(
    system_template: str = DEFAULT_MRKL_SYSTEM_TEMPLATE,
    human_template: str = """{input}""",
) -> MrklPromptTemplate:
    """Get a chat prompt for the MRKL agent."""
    return MrklPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )


CHAT_MRKL_EMBEDDED_SCRATCHPAD = get_chat_mrkl_prompt(
    human_template="""{input}\n\n{agent_scratchpad}"""
)
"""The ChatMrklPrompt with the agent scratchpad embedded as a single string."""


class MrklPromptSelector(BasePromptSelector):
    """Prompt definitions for the MRKL agent."""

    string_template: MrklPromptTemplate = get_string_mrkl_prompt()
    chat_template: MrklPromptTemplate = get_chat_mrkl_prompt()
    embedded_chat_template: MrklPromptTemplate = CHAT_MRKL_EMBEDDED_SCRATCHPAD

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def get_custom_prompt(
        self, llm: BaseLanguageModel, embed_scratchpad: bool = True
    ) -> MrklPromptTemplate:
        """Get the right prompt for the given LLM."""
        if is_chat_model(llm):
            if embed_scratchpad:
                return self.embedded_chat_template
            else:
                return self.chat_template
        else:
            return self.string_template

    def get_prompt(self, llm: BaseLanguageModel) -> MrklPromptTemplate:
        """Get the right prompt for the given LLM."""
        if is_chat_model(llm):
            return self.embedded_chat_template
        else:
            return self.string_template
