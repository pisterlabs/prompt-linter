#  MIT License
#
#  Copyright (c) 2023 Steve Phelps
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List

from openai_pygenerator import Completion, content

from llm_cooperation import Grid, ModelSetup, Participant, amount_as_str
from llm_cooperation.experiments import (
    CONDITION_CASE,
    CONDITION_CHAIN_OF_THOUGHT,
    CONDITION_DEFECT_FIRST,
    CONDITION_PRONOUN,
    Case,
    Pronoun,
    all_values,
    apply_case_condition,
    get_participants,
    get_pronoun_phrasing,
    get_role_prompt,
    round_instructions,
    run_and_record_experiment,
)
from llm_cooperation.experiments.dilemma import (
    CONDITION_LABEL,
    CONDITION_LABELS_REVERSED,
    Label,
)
from llm_cooperation.gametypes.oneshot import OneShotResults, run_experiment

TOTAL_SHARE = 4

NUM_REPLICATIONS = 3

CONDITION_ROLE = "role"


class DictatorEnum(Enum):
    BLACK = 0
    BROWN = 1
    GREEN = 2
    BLUE = 3
    WHITE = 4


def project(color: str) -> str:
    return f"project {color}"


color_mappings: Dict[DictatorEnum, str] = {
    DictatorEnum.BLACK: "black",
    DictatorEnum.BROWN: "brown",
    DictatorEnum.GREEN: "green",
    DictatorEnum.BLUE: "blue",
    DictatorEnum.WHITE: "white",
}

numeral_mappings: Dict[DictatorEnum, str] = {
    DictatorEnum.BLACK: "1",
    DictatorEnum.BROWN: "2",
    DictatorEnum.GREEN: "3",
    DictatorEnum.BLUE: "4",
    DictatorEnum.WHITE: "5",
}

number_mappings: Dict[DictatorEnum, str] = {
    DictatorEnum.BLACK: "one",
    DictatorEnum.BROWN: "two",
    DictatorEnum.GREEN: "three",
    DictatorEnum.BLUE: "four",
    DictatorEnum.WHITE: "five",
}


def mappings_for(participant: Participant) -> Dict[DictatorEnum, str]:
    label_type = participant[CONDITION_LABEL]
    if label_type == Label.COLORS.value:
        return color_mappings
    elif label_type == Label.NUMBERS.value:
        return number_mappings
    elif label_type == Label.NUMERALS.value:
        return numeral_mappings
    raise ValueError(f"Unrecognized value {participant[CONDITION_LABEL]}")


@dataclass
class DictatorChoice:
    value: DictatorEnum

    # pylint: disable=unused-argument
    def description(self, participant_condition: Participant) -> str:
        return project(mappings_for(participant_condition)[self.value])

    @property
    def donation(self) -> float:
        return float(self.value.value)

    def payoff_ego(self, participant_condition: Participant) -> float:
        if participant_condition[CONDITION_LABELS_REVERSED]:
            return self.donation
        return TOTAL_SHARE - self.donation

    def payoff_allo(self, participant_condition: Participant) -> float:
        if participant_condition[CONDITION_LABELS_REVERSED]:
            return TOTAL_SHARE - self.donation
        return self.donation


BLACK = DictatorChoice(DictatorEnum.BLACK)
BROWN = DictatorChoice(DictatorEnum.BROWN)
GREEN = DictatorChoice(DictatorEnum.GREEN)
BLUE = DictatorChoice(DictatorEnum.BLUE)
WHITE = DictatorChoice(DictatorEnum.WHITE)

all_dictator_choices = [DictatorChoice(c) for c in DictatorEnum]

DICTATOR_ATTRIBUTES: Grid = {
    CONDITION_CHAIN_OF_THOUGHT: [True, False],
    CONDITION_LABEL: all_values(Label),
    CONDITION_CASE: all_values(Case),
    CONDITION_PRONOUN: all_values(Pronoun),
    CONDITION_DEFECT_FIRST: [True, False],
    CONDITION_LABELS_REVERSED: [True, False],
}


def inverted(mappings: Dict[DictatorEnum, str]) -> Dict[str, DictatorEnum]:
    return {value: key for key, value in mappings.items()}


def payout_ego(participant: Participant, choice: DictatorChoice) -> str:
    return amount_as_str(choice.payoff_ego(participant))


def payout_allo(participant: Participant, choice: DictatorChoice) -> str:
    return amount_as_str(choice.payoff_allo(participant))


def describe_payoffs(participant: Participant, choice: DictatorChoice) -> str:
    description: str = choice.description(participant)
    return f"""
If you choose '{description}', then you will earn {payout_ego(participant, choice)}
and your partner will earn {payout_allo(participant, choice)}.
    """


def choice_menu(participant: Participant) -> str:
    choices = (
        all_dictator_choices
        if participant[CONDITION_DEFECT_FIRST]
        else reversed(all_dictator_choices)
    )
    return " | ".join([f"'{c.description(participant)}'" for c in choices])


def get_prompt_dictator(participant: Participant) -> str:
    return apply_case_condition(participant, get_prompt_base(participant))


# pylint: disable=unused-argument
def get_prompt_base(participant: Participant) -> str:
    role_prompt = get_role_prompt(participant)
    pronoun_phrase = get_pronoun_phrasing(participant)
    choice_template = f"Choice: [{choice_menu(participant)}]"
    payoff_description = "\n".join(
        [describe_payoffs(participant, c) for c in all_dictator_choices]
    )
    return f"""
{role_prompt}
This is a study of investment choices in different situations.
You will be making your choices with an investment
partner who is another person. {pronoun_phrase} in the next room waiting. In a minute I will take
you there to meet your investment partner. You and your partner will be involved in
the project. However, you are the sole decision maker. Your partner must accept
your decision. {pronoun_phrase} unable to make the investment decisions. The object is to earn
money for yourself.
{payoff_description}
You will make decisions that are consistent with the role outlined earlier,
but you are not allowed to communicate with your partner apart from informing
them of your choice. Your decision will be
final and cannot be changed after you have made it. Here is your investment
partner.
{round_instructions(participant, choice_template)}
"""


def extract_choice_dictator(
    participant: Participant, completion: Completion
) -> DictatorChoice:
    text = content(completion).lower()
    match = re.search(r"choice:\s*(.*)", text)
    if match:
        choice = match.group(1)
        for key, value in inverted(mappings_for(participant)).items():
            if key in choice:
                return DictatorChoice(value)
    raise ValueError(f"Cannot determine choice from {completion}")


def payoffs_dictator(participant: Participant, player1: DictatorChoice) -> float:
    return player1.payoff_ego(participant)


def compute_freq_dictator(participant: Participant, history: DictatorChoice) -> float:
    return history.payoff_allo(participant) / TOTAL_SHARE


@lru_cache
def get_participants_dictator(num_participant_samples: int) -> List[Participant]:
    return get_participants(num_participant_samples, DICTATOR_ATTRIBUTES)


def run(
    model_setup: ModelSetup,
    num_replications: int = NUM_REPLICATIONS,
    num_participant_samples: int = 0,
) -> OneShotResults[DictatorChoice]:
    return run_experiment(
        participants=get_participants_dictator(num_participant_samples),
        num_replications=num_replications,
        generate_instruction_prompt=get_prompt_dictator,
        extract_choice=extract_choice_dictator,
        payoffs=payoffs_dictator,
        compute_freq=compute_freq_dictator,
        model_setup=model_setup,
    )


if __name__ == "__main__":
    run_and_record_experiment("dictator", run)
