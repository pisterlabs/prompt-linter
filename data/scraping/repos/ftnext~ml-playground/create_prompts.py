import argparse
from collections.abc import Generator
from typing import Literal

import jsonlines
from datasets import load_dataset
from langchain.prompts import HumanMessagePromptTemplate

from custom_types import Conll03Example, Example
from instructions import get_instruction


def load_test_set() -> Generator[Conll03Example, None, None]:
    conll03 = load_dataset("conll2003")
    for example in conll03["test"]:
        yield Conll03Example(
            id=example["id"],
            tokens=example["tokens"],
            ner_tags=example["ner_tags"],
        )


def build_prompt_template(instruction_number: Literal[1, 2, 3, 4, 5]) -> str:
    instruction = get_instruction(instruction_number)
    # Zero shot https://github.com/FreedomIntelligence/Evaluation-of-ChatGPT-on-Information-Extraction/blob/bda6894cffd041ff629ba1c2c9473e757ba217eb/1_NER/ner_test_with_api.py#L71
    template = (
        instruction
        + """\
Given sentence:
"{sentence}"\
"""
    )
    return HumanMessagePromptTemplate.from_template(template)


def create_prompts(
    instruction_number: Literal[1, 2, 3, 4, 5]
) -> Generator[Example, None, None]:
    prompt_template = build_prompt_template(instruction_number)
    examples = load_test_set()
    for example in examples:
        sentence = " ".join(example["tokens"])
        yield Example(
            **example,
            prompt=prompt_template.format(sentence=sentence).content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_jsonl")
    parser.add_argument(
        "--instruction_number", type=int, choices=[1, 2, 3, 4, 5], default=2
    )
    args = parser.parse_args()

    with jsonlines.open(args.output_jsonl, "w") as writer:
        writer.write_all(create_prompts(args.instruction_number))
