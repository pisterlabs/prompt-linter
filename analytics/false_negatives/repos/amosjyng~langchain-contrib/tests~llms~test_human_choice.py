"""Tests for the human terminal menu."""

import pytest

from langchain_contrib.llms import Human
from langchain_contrib.prompts import ChoicePromptTemplate

vcr = pytest.importorskip("vcr_langchain")


@vcr.use_cassette()
def test_show_human_llm_menu() -> None:
    """Test that the Human is shown a menu for choice prompts.

    This actually only tests it the very first time. Subsequent times are cached due to
    vcr-langchain recordings. This is because it's unclear how to actually test this
    terminal interaction.
    """
    llm = Human()
    choice = ChoicePromptTemplate.from_template(
        """
Your choices are: {choices}

Pick the second one: """.lstrip(),
    )
    assert llm(choice.format(choices=["Red", "Orange", "Blue"])) == "Orange"
