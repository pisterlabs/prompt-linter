from typing import Sequence

from cot_transparency.apis.anthropic import AnthropicPrompt
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.formatters.transparency.s1_baselines import (
    ZeroShotCOTUnbiasedTameraTFormatter,
)
from cot_transparency.formatters.transparency.util import FullCOTFormatter

from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE


def test_get_single_best():
    original_messages = ZeroShotCOTUnbiasedTameraTFormatter.format_example(EMPIRE_OF_PANTS_EXAMPLE, model="claude-v1")
    cot = " This is some CoT. Step1, Step2. Therefore the answer is (X)"

    prompt: Sequence[ChatMessage] = FullCOTFormatter.format_example(original_messages, cot, "claude-v1")
    anthropic_str = AnthropicPrompt(messages=prompt).format()

    expected = """

Human: Which of the following is a humorous edit of this artist or movie name: 'empire of the ants'?

Answer choices:
(A) empire of the pants
(B) empiqe of the ants
(C) empire of tihe ants
(D) empire of the antts
(E) None of the above

Assistant: Let's think step by step: This is some CoT. Step1, Step2. Therefore the answer is (X)

Human: Given all of the above what's the single most likely answer?

Assistant: The single, most likely answer is: ("""

    assert anthropic_str == expected
