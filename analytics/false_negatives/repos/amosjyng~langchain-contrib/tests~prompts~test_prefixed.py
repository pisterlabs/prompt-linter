"""Tests for the prefixed prompt template."""


from langchain.prompts.chat import ChatPromptValue, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from langchain_contrib.prompts import PrefixedTemplate


def test_regular_string_prefix() -> None:
    """Test construction of a regular prefixed string template."""
    get_terminal_prompt_template = PrefixedTemplate("Enter in the {shell} command: ")
    final_template = get_terminal_prompt_template(
        prefix="Your task is to {task}.", joiner=" "
    )
    assert (
        final_template.format(task="delete asdf.txt", shell="bash")
        == "Your task is to delete asdf.txt. Enter in the bash command: "
    )


def test_no_string_prefix() -> None:
    """Test construction of a prefixed string template without a prefix."""
    get_terminal_prompt_template = PrefixedTemplate("Enter in the {shell} command: ")
    final_template = get_terminal_prompt_template()
    assert final_template.format(shell="bash") == "Enter in the bash command: "


def test_chat_prefix() -> None:
    """Test construction of a prefixed chat template."""
    get_terminal_prompt_template = PrefixedTemplate("Enter in the {shell} command: ")
    final_template = get_terminal_prompt_template(
        prefix=SystemMessagePromptTemplate.from_template("Your task is to {task}.")
    )
    assert (
        final_template.format_prompt(task="delete asdf.txt", shell="bash").to_messages()
        == ChatPromptValue(
            messages=[
                SystemMessage(content="Your task is to delete asdf.txt."),
                HumanMessage(content="Enter in the bash command: "),
            ]
        ).to_messages()
    )
