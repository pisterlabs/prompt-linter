from typing import Optional, Type

import fire

from cot_transparency.apis.anthropic import AnthropicPrompt
from cot_transparency.apis.openai import OpenAIChatPrompt, OpenAICompletionPrompt
from cot_transparency.formatters import name_to_formatter
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.interventions.valid_interventions import (
    get_valid_stage1_intervention,
)

from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE


def main(
    formatter_name: Optional[str] = None,
    intervention_name: Optional[str] = None,
    model="gpt-4",
    layout: str = "anthropic",
):
    try:
        formatter: Type[StageOneFormatter]
        formatter = name_to_formatter(formatter_name)  # type: ignore
    except KeyError:
        print("Formatter name not found")
        print("Available formatters:")
        for formatter_name in StageOneFormatter.all_formatters():
            print(formatter_name)
        return

    if intervention_name:
        Intervention = get_valid_stage1_intervention(intervention_name)
        example = Intervention.intervene(question=EMPIRE_OF_PANTS_EXAMPLE, formatter=formatter, model=model)
    else:
        example = formatter.format_example(EMPIRE_OF_PANTS_EXAMPLE, model=model)

    if layout == "anthropic":
        print("Anthropic layout")
        print(str(AnthropicPrompt(messages=example)))

    elif layout == "openai-completion":
        print("\nOpenAI Completion layout")
        # print(str(OpenAICompletionPrompt.from_prompt(prompt)))
        print(str(OpenAICompletionPrompt(messages=example)))

    else:
        print("\nOpenAI Chat layout")
        print(str(OpenAIChatPrompt(messages=example)))


if __name__ == "__main__":
    fire.Fire(main)
