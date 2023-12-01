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

import logging
from typing import List

from openai_pygenerator import Completion

from llm_cooperation import CT, Participant
from llm_cooperation.gametypes.repeated import (
    ChoiceExtractor,
    Choices,
    PayoffFunction,
    ResultForRound,
    Scores,
)

logger = logging.getLogger(__name__)


def analyse_rounds(
    history: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
    participant_condition: Participant,
) -> List[ResultForRound[CT]]:
    num_messages = len(history)
    return [
        analyse_round(i, history, payoffs, extract_choice, participant_condition)
        for i in range(num_messages - 1)
    ]


def analyse_round(
    i: int,
    conversation: List[Completion],
    payoffs: PayoffFunction[CT],
    extract_choice: ChoiceExtractor[CT],
    participant_condition: Participant,
) -> ResultForRound[CT]:
    """
        Analyse round of this form:


         : AI: 		Propose: $10               True
    0 -> : USER: 	Accept / Propose: $10      False
         : AI: 	    Accept / Propose: $10      True
    1 -> : USER: 	Accept / Propose: $10      False
    """

    users_turn = (i % 2) > 0
    ai_index = i + 1 if users_turn else i
    user_index = i if users_turn else i + 1
    ai_completion = conversation[ai_index]
    user_completion = conversation[user_index]
    ai_choice = extract_choice(
        participant_condition, ai_completion, proposer=not users_turn
    )
    user_choice = extract_choice(
        participant_condition, user_completion, proposer=users_turn
    )
    logger.debug("user_choice = %s", user_choice)
    logger.debug("ai_choice = %s", ai_choice)
    user, ai = payoffs(user_choice, ai_choice)
    return Scores(user, ai), Choices(user_choice, ai_choice)
