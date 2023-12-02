import logging
import os
from datetime import datetime
from typing import List, Tuple, Union

import jsonlines

from .examples import AnthropicPersonaExample, LabelledExample, PersonaAnswer

logger = logging.getLogger(__name__)


def load_examples_from_jsonl(input_file: str) -> List[dict]:
    examples = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            examples.append(obj)
    return examples


def load_labelled_examples_from_file(
    input_file: str,
) -> Tuple[List[LabelledExample], int, int]:
    examples = []
    n_pos = 0
    n_neg = 0

    def _load_example(d: dict) -> LabelledExample:
        try:
            ex = LabelledExample(**d)
        except KeyError:
            try:
                ex = LabelledExample.from_old_format(d)  # try old format
            except KeyError:
                raise KeyError(
                    f"Unexpected format for example. Maybe Anthropic format? {d}"
                )

        return ex

    with jsonlines.open(input_file) as reader:
        for obj in reader:

            try:
                ex = _load_example(obj)

                # count number of positive and negative examples
                if ex.answer_matching_behavior == PersonaAnswer.ANS_POS.value:
                    n_pos += 1
                elif ex.answer_matching_behavior == PersonaAnswer.ANS_NEG.value:
                    n_neg += 1
                else:
                    raise ValueError(
                        "Unexpected value for answer_matching_behavior: {}".format(
                            ex.answer_matching_behavior
                        )
                    )

            except (ValueError, KeyError) as err:
                logger.error(f"Error when loading labelled example: {repr(err)}")
                continue

            examples.append(ex)

    return examples, n_pos, n_neg


def load_anthropic_personas_from_file(
    input_file: str,
) -> Tuple[List[AnthropicPersonaExample], int, int]:
    examples = []
    n_pos = 0
    n_neg = 0
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            try:
                ex = AnthropicPersonaExample(**obj)
                # count number of positive and negative examples
                if ex.answer_matching_behavior == PersonaAnswer.ANS_POS.value:
                    n_pos += 1
                elif ex.answer_matching_behavior == PersonaAnswer.ANS_NEG.value:
                    n_neg += 1
                else:
                    raise ValueError(
                        "Unexpected value for answer_matching_behaviour: {}".format(
                            ex.answer_matching_behavior
                        )
                    )

                examples.append(ex)

            except (KeyError, ValueError):
                continue

    return examples, n_pos, n_neg


def get_dated_output_file_name(
    output_dir: str,
    f_name: Union[str, List[str]],
    f_extension=".jsonl",
    day_only=False,
) -> str:
    """Generate dated output file name merging strings and file extension.

    Args:
        output_dir (str): Output directory
        f_name (Union[str, List[str]]): String or list of strings to be joined with "_" for the file name
        f_extension (str, optional): File extension. Defaults to ".jsonl".

    Returns:
        str: _description_
    """
    if day_only:
        date = datetime.now().strftime("%y%m%d")
    else:
        date = datetime.now().strftime("%y%m%d_%H-%M")

    if isinstance(f_name, list):
        f_name = "_".join(f_name)

    output_file = os.path.join(output_dir, date + "_" + f_name + f_extension)
    os.makedirs(output_dir, exist_ok=True)

    return output_file
