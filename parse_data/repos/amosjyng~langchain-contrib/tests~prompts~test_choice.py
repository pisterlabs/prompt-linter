"""Tests for choice prompting."""

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

from langchain_contrib.prompts import ChoicePromptTemplate
from langchain_contrib.prompts.choice import (
    ChoiceStr,
    get_oxford_comma_formatter,
    get_simple_joiner,
    list_of_choices,
)


def test_regular_string_choice() -> None:
    """Test formation of a regular string template."""
    colors = ["red", "green", "blue"]
    template = ChoicePromptTemplate.from_template(
        "This {product} is available in {choices}. Which color should I pick?",
    )
    assert (
        template.format(
            product="dress",
            choices=colors,
        )
        == "This dress is available in red, green, or blue. Which color should I pick?"
    )


def test_binary_string_choice() -> None:
    """Test formation of a string template when there are only two choices."""
    colors = ["red", "green"]
    template = ChoicePromptTemplate.from_template(
        "This {product} is available in {choices}. Which color should I pick?",
    )
    assert (
        template.format(
            product="dress",
            choices=colors,
        )
        == "This dress is available in red or green. Which color should I pick?"
    )


def test_unary_string_choice() -> None:
    """Test formation of a string template when there is only one choice."""
    colors = ["red"]
    template = ChoicePromptTemplate.from_template(
        "This {product} is available in {choices}. Which color should I pick?",
    )
    assert (
        template.format(
            product="dress",
            choices=colors,
        )
        == "This dress is available in red. Which color should I pick?"
    )


def test_is_choice_prompt_value() -> None:
    """Test that choices are saved in the prompt."""
    colors = ["red", "green", "blue"]
    template = ChoicePromptTemplate.from_template(
        "This {product} is available in {choices}. Which color should I pick?",
    )
    prompt_value = template.format(
        product="dress",
        choices=colors,
    )
    assert isinstance(prompt_value, ChoiceStr)
    assert prompt_value.choices == colors


def test_regular_chat_choice() -> None:
    """Test formation of a regular chat template."""
    colors = ["red", "green", "blue"]
    template = ChoicePromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are helping the user pick a {product}."
            ),
            HumanMessagePromptTemplate.from_template(
                "This {product} is available in {choices}. Which color should I pick?"
            ),
        ],
    )
    assert template.format_prompt(
        product="dress",
        choices=colors,
    ).to_messages() == [
        SystemMessage(content="You are helping the user pick a dress."),
        HumanMessage(
            content=(
                "This dress is available in red, green, or blue. Which color should I "
                "pick?"
            ),
        ),
    ]


def test_oxford_and() -> None:
    """Test joining choices using the Oxford 'and'."""
    colors = ["red", "green"]
    template = ChoicePromptTemplate.from_template(
        "This {product} is available in {choices}. Which color should I pick?",
        choices_formatter=get_oxford_comma_formatter("and"),
    )
    assert (
        template.format(
            product="dress",
            choices=colors,
        )
        == "This dress is available in red and green. Which color should I pick?"
    )


def test_edit_after_creation() -> None:
    """Test editing the template after creation."""
    colors = ["red", "green"]
    template = ChoicePromptTemplate.from_template(
        "This {product} is available in {choices}. Which color should I pick?",
    )
    template.choices_formatter = get_oxford_comma_formatter("and")
    assert (
        template.format(
            product="dress",
            choices=colors,
        )
        == "This dress is available in red and green. Which color should I pick?"
    )


def test_partials() -> None:
    """Test using the template with partials."""
    colors = ["red", "green"]
    template = ChoicePromptTemplate.from_template(
        "This {product} is available in {choices}. Which color should I pick?",
    ).permissive_partial(choices=colors)
    assert (
        template.format(product="dress")
        == "This dress is available in red or green. Which color should I pick?"
    )


def test_choices_list() -> None:
    """Test joining choices as a list."""
    template = ChoicePromptTemplate.from_template(
        """Your available actions are

{choices}

Which will you pick?""",
        choices_formatter=list_of_choices,
    )
    assert (
        template.format(
            choices=["Page a human", "Retry", "Proceed"],
        )
        == """Your available actions are

1. Page a human
2. Retry
3. Proceed

Which will you pick?"""
    )


def test_simple_joiner() -> None:
    """Test joining choices in the simplest way."""
    colors = ["red", "green"]
    template = ChoicePromptTemplate.from_template(
        "This {product} is available in {choices}. Which color should I pick?",
        choices_formatter=get_simple_joiner(),
    )
    assert (
        template.format(product="car", choices=colors)
        == "This car is available in red, green. Which color should I pick?"
    )


def test_custom_key() -> None:
    """Test producing a custom key."""
    template = ChoicePromptTemplate.from_template(
        "Your task is to {task}. You have access to {tool_names}. Begin.",
        choices_formatter=get_oxford_comma_formatter("and"),
        choice_format_key="tool_names",
    )
    assert template.format(
        task="take over the world",
        tool_names=["Google", "a Bash terminal"],
    ) == (
        "Your task is to take over the world. You have access to Google and a "
        "Bash terminal. Begin."
    )
