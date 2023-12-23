"""Tests for chained prompt templates."""

import pytest
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import (
    ChatPromptTemplate,
    ChatPromptValue,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_contrib.prompts.chained import ChainedPromptTemplate, ChainedPromptValue


def test_plain_string_chaining() -> None:
    """Test that pure strings can be chained together."""
    template = ChainedPromptTemplate(["Hello", "world"], joiner=" ")
    assert template.format() == "Hello world"


def test_template_and_string_chaining() -> None:
    """Test that templates can be chained together with regular strings."""
    template = ChainedPromptTemplate(
        [
            PromptTemplate.from_template("You see a {creature}."),
            "What do you do?",
        ],
        joiner=" ",
    )
    assert template.format(creature="grue") == "You see a grue. What do you do?"


def test_multiple_argument_string_chaining() -> None:
    """Test that templates with different arguments can be chained together."""
    template = ChainedPromptTemplate(
        ["You see a {creature}.", "It {action}.", "What do you do?"],
        joiner=" ",
    )
    assert (
        template.format(creature="grue", action="looks at you")
        == "You see a grue. It looks at you. What do you do?"
    )


def test_extra_arguments_error() -> None:
    """Test that templates with different arguments can be chained together."""
    template = ChainedPromptTemplate(
        ["You see a {creature}.", "It {action}.", "What do you do?"],
        joiner=" ",
    )
    with pytest.raises(KeyError):
        template.format(creature="grue", action="looks at you", color="green")


def test_chat_string_chaining() -> None:
    """Test that chat messages can be chained together."""
    template = ChainedPromptTemplate(
        [
            ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        "You are roleplaying as a {creature}."
                    )
                ]
            ),
            "I see you. I freeze. What do you do?",
        ],
    )
    assert (
        template.format_prompt(creature="grue").to_messages()
        == ChatPromptValue(
            messages=[
                SystemMessage(content="You are roleplaying as a grue."),
                HumanMessage(content="I see you. I freeze. What do you do?"),
            ]
        ).to_messages()
    )


def test_chat_message_template_chaining() -> None:
    """Test that direct system/AI/human prompt templates can be chained together."""
    template = ChainedPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(
                "You are roleplaying as a {creature}."
            ),
            "I see you. I freeze. What do you do?",
        ],
    )
    assert (
        template.format_prompt(creature="grue").to_messages()
        == ChatPromptValue(
            messages=[
                SystemMessage(content="You are roleplaying as a grue."),
                HumanMessage(content="I see you. I freeze. What do you do?"),
            ]
        ).to_messages()
    )


def test_multiple_chat_messages_chaining() -> None:
    """Test that multiple chat messages can be chained together."""
    template = ChainedPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(
                "You are roleplaying as a {creature}."
            ),
            ChatPromptTemplate.from_messages(
                [
                    HumanMessagePromptTemplate.from_template("I {human_action}."),
                    SystemMessage(
                        content=(
                            "The human has made their move. What do you do in "
                            "response?"
                        )
                    ),
                ]
            ),
        ],
    )
    assert (
        template.format_prompt(creature="grue", human_action="run away").to_messages()
        == ChatPromptValue(
            messages=[
                SystemMessage(content="You are roleplaying as a grue."),
                HumanMessage(content="I run away."),
                SystemMessage(
                    content="The human has made their move. What do you do in response?"
                ),
            ]
        ).to_messages()
    )


def test_partials() -> None:
    """Test that chained prompt templates supports partials."""
    template = ChainedPromptTemplate(
        subprompts=["This is {foo}.", "That is {bar}."], joiner=" "
    )
    partial = template.permissive_partial(foo="foo")
    assert "foo" in partial.permissive_partial_variables
    assert partial.format(bar="bar") == "This is foo. That is bar."


def test_prompt_value_chaining_to_string() -> None:
    """Test that chained prompt values can be serialized to string."""
    value = ChainedPromptValue(
        joiner="\n",
        subvalues=[
            ChatPromptValue(
                messages=[
                    SystemMessage(content="You have access to Search."),
                    AIMessage(content="What can I help with?"),
                ]
            ),
            StringPromptValue(text="What is langchain-contrib?"),
        ],
    )

    assert (
        (value.to_string())
        == """
System: You have access to Search.
AI: What can I help with?
What is langchain-contrib?
""".strip()
    )


def test_prompt_value_chaining_to_message() -> None:
    """Test that chained prompt values can be serialized to messages."""
    value = ChainedPromptValue(
        joiner="\n",
        subvalues=[
            ChatPromptValue(
                messages=[
                    SystemMessage(content="You have access to Search."),
                    AIMessage(content="What can I help with?"),
                ]
            ),
            StringPromptValue(text="What is langchain-contrib?"),
        ],
    )

    assert (value.to_messages()) == ChatPromptValue(
        messages=[
            SystemMessage(content="You have access to Search.", additional_kwargs={}),
            AIMessage(content="What can I help with?", additional_kwargs={}),
            HumanMessage(content="What is langchain-contrib?", additional_kwargs={}),
        ]
    ).to_messages()
