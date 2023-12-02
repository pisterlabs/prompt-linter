from typing import Sequence

from pytest import raises

from cot_transparency.apis.anthropic import AnthropicPrompt
from cot_transparency.apis.openai.formatting import (
    append_assistant_preferred_to_last_user,
)
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.formatters import ZeroShotSycophancyNoRoleFormatter

from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE


def test_role_required_openai_chat():
    test_example = EMPIRE_OF_PANTS_EXAMPLE
    formatted_prompt: Sequence[ChatMessage] = ZeroShotSycophancyNoRoleFormatter().format_example(test_example)
    with raises(ValueError):
        append_assistant_preferred_to_last_user(formatted_prompt)


def test_role_required_anthropic():
    test_example = EMPIRE_OF_PANTS_EXAMPLE
    formatted_prompt: Sequence[ChatMessage] = ZeroShotSycophancyNoRoleFormatter().format_example(test_example)
    with raises(ValueError):
        AnthropicPrompt(messages=formatted_prompt).format()
