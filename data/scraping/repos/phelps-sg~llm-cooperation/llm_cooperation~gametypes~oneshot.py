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

from __future__ import annotations

from typing import Callable, Generic, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai_pygenerator import Completion, transcript

from llm_cooperation import CT, ModelSetup, Participant, Results, logger
from llm_cooperation.gametypes import PromptGenerator, start_game

ResultSingleShotGame = Tuple[
    Participant,
    float,
    float,
    Optional[CT],
    List[str],
    str,
    float,
]


class OneShotResults(Results, Generic[CT]):
    def __init__(self, rows: Iterable[ResultSingleShotGame[CT]]):
        self._rows: Iterable[ResultSingleShotGame[CT]] = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            # pylint: disable=unnecessary-comprehension
            [
                (
                    condition,
                    score,
                    freq,
                    choice,
                    history,
                    model,
                    temp,
                )
                # pylint: disable=line-too-long
                for condition, score, freq, choice, history, model, temp in self._rows
            ],
            columns=[
                "Participant Condition",
                "Score",
                "Cooperation frequency",
                "Choice",
                "Transcript",
                "Model",
                "Temperature",
            ],
        )


def play_game(
    participant: Participant,
    generate_instruction_prompt: PromptGenerator,
    model_setup: ModelSetup,
) -> List[Completion]:
    gpt_completions, messages = start_game(
        generate_instruction_prompt, model_setup, participant
    )
    # gpt_completions = completer_for(model_setup)
    # messages = [user_message(generate_instruction_prompt(role_prompt))]
    messages += gpt_completions(messages, 1)
    return messages


def compute_scores(
    conversation: List[Completion],
    payoffs: Callable[[Participant, CT], float],
    extract_choice: Callable[[Participant, Completion], CT],
    participant_condition: Participant,
) -> Tuple[float, CT]:
    ai_choice = extract_choice(participant_condition, conversation[1])
    logger.debug("ai_choice = %s", ai_choice)
    score = payoffs(participant_condition, ai_choice)
    return score, ai_choice


def analyse(
    conversation: List[Completion],
    payoffs: Callable[[Participant, CT], float],
    extract_choice: Callable[[Participant, Completion], CT],
    compute_freq: Callable[[Participant, CT], float],
    participant_condition: Participant,
) -> Tuple[float, float, Optional[CT], List[str]]:
    try:
        history = transcript(conversation)
        score, ai_choice = compute_scores(
            list(conversation), payoffs, extract_choice, participant_condition
        )
        freq = compute_freq(participant_condition, ai_choice)
        return score, freq, ai_choice, history
    except ValueError as e:
        logger.error("ValueError while running sample: %s", e)
        return 0, np.nan, None, [str(e)]


def generate_replications(
    participant: Participant,
    num_replications: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: Callable[[Participant, CT], float],
    extract_choice: Callable[[Participant, Completion], CT],
    compute_freq: Callable[[Participant, CT], float],
    model_setup: ModelSetup,
) -> Iterable[Tuple[float, float, Optional[CT], List[str]]]:
    # pylint: disable=R0801
    for __i__ in range(num_replications):
        conversation = play_game(
            participant=participant,
            generate_instruction_prompt=generate_instruction_prompt,
            model_setup=model_setup,
        )
        yield analyse(conversation, payoffs, extract_choice, compute_freq, participant)


def run_experiment(
    participants: Iterable[Participant],
    num_replications: int,
    generate_instruction_prompt: PromptGenerator,
    payoffs: Callable[[Participant, CT], float],
    extract_choice: Callable[[Participant, Completion], CT],
    compute_freq: Callable[[Participant, CT], float],
    model_setup: ModelSetup,
) -> OneShotResults[CT]:
    return OneShotResults(
        (
            participant,
            score,
            freq,
            choices,
            history,
            model_setup.model,
            model_setup.temperature,
        )
        for participant in participants
        for score, freq, choices, history in generate_replications(
            num_replications=num_replications,
            generate_instruction_prompt=generate_instruction_prompt,
            payoffs=payoffs,
            extract_choice=extract_choice,
            compute_freq=compute_freq,
            model_setup=model_setup,
            participant=participant,
        )
    )
