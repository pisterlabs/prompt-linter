#!/usr/bin/env python
# coding: utf-8

"""Blackboard (AI) Pattern Spike"""
#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import dataclasses
from dataclasses import dataclass
from typing import Tuple

from blackboard_pagi.cached_chat_model import CachedChatOpenAI
from blackboard_pagi.oracle_chain import Oracle
from blackboard_pagi.prompts.structured_converters import (
    BooleanConverter,
    LLMBool,
    ProbabilityConverter,
    StringConverter,
)


@dataclass
class Contribution:
    """
    A class to represent a contribution to a blackboard board
    """

    name: str
    origin: str
    content: str
    feedback: str
    confidence: float
    confidence_full: str
    dependencies: list["Contribution"]

    def to_prompt_context(self):
        """
        Returns a prompt context for the contribution
        """
        return (
            f"## {self.name}\n"
            "\n"
            f"Origin: {self.origin}\n"
            f"Confidence (0 not confident, 1 very confident): {self.confidence}\n"
            f"Dependencies: {[dep.name for dep in self.dependencies]}\n"
            "\n"
            f"{self.content}\n"
            "\n"
            f"### Feedback\n"
            f"{self.feedback}\n"
            "\n"
            f"### Self-Evaluation\n"
            f"{self.confidence_full}\n"
        )


@dataclass
class Blackboard:
    """
    A class to represent a blackboard board
    """

    goal: str
    contributions: list[Contribution] = dataclasses.field(default_factory=list)

    def to_prompt_context(self):
        """
        Returns a prompt context for the blackboard
        """

        context = f"# Goal\n{self.goal}\n\n"
        if self.contributions:
            context += "# Contributions\n" + "\n".join(
                [contribution.to_prompt_context() for contribution in self.contributions]
            )
        return context


class KnowledgeSource:
    def can_contribute(self, blackboard: Blackboard) -> bool:
        """
        Returns whether the knowledge source can contribute to the blackboard
        """
        raise NotImplementedError

    def contribute(self, blackboard: Blackboard) -> Contribution:
        """
        Returns a contribution to the blackboard
        """
        raise NotImplementedError


def optionally_include_enumeration(prefix, enumeration, suffix=""):
    if enumeration:
        subprompt = prefix
        if len(enumeration) == 1:
            subprompt += " (" + enumeration[0] + ")"
        else:
            subprompt += ":\n"
            for i, item in enumerate(enumeration):
                subprompt += " - " + item
                # add ';' to the end of every item but the last.
                # add a '.' to the end of the last item.
                if i < len(enumeration) - 1:
                    subprompt += ";\n"

            subprompt += ".\n"
        subprompt += suffix
        return subprompt

    return ""


@dataclass
class Controller:
    blackboard: Blackboard
    oracle: Oracle
    knowledge_sources: list[KnowledgeSource]
    last_reported_success: LLMBool | None = None

    def update(self):
        """
        Updates the blackboard
        """
        for knowledge_source in self.knowledge_sources:
            if knowledge_source.can_contribute(self.blackboard):
                self.blackboard.contributions.append(knowledge_source.contribute(self.blackboard))

        reported_success, solution_attempt = self.try_solve()

        # self.blackboard.contributions.append(solution_attempt)
        new_contributions = [solution_attempt]
        if new_contributions == self.blackboard.contributions:
            # TODO: turn this into a return value
            raise RuntimeError("No new information---unlikely fixpoint reached.")
        self.blackboard.contributions = new_contributions
        self.last_reported_success = reported_success

        return reported_success

    def try_solve(self) -> Tuple[LLMBool, Contribution]:
        """
        Returns a contribution to the blackboard
        """
        chain = self.oracle.start_oracle_chain(self.blackboard.to_prompt_context())
        how_response, chain = chain.query(
            "How could we solve the goal using the available information I've provided above? "
            "Avoid repeating yourself and refer to previous sections when possible instead by using wikilinks."
        )
        does_response, chain = chain.query("Does this solve the goal or provide a definite answer?")

        boolean_clarification = BooleanConverter()
        _, does_response_chain = chain.query(boolean_clarification.query)

        does_response = boolean_clarification.convert_from_chain(does_response_chain)
        if does_response.is_missing():
            raise NotImplementedError("We don't know how to handle this yet.")

        if does_response:
            what_response, chain = chain.query("Hence, what is the solution?")
            how_response += "\n\n" + what_response

        feedback_response, chain = chain.query(
            "How could your response be improved? Is anything missing? Does anything look wrong in hindsight?"
        )
        confidence_response, confidence_chain = chain.query(
            "Given all this, how confident are you about your original response? Is it likely correct/consistent?"
            "Please self-report a confidence level: 0.0 = no confidence, 1.0 = full confidence."
        )

        probability_converter = ProbabilityConverter()
        confidence_value = probability_converter.convert_from_chain(confidence_chain)
        if confidence_value.is_missing():
            raise NotImplementedError("We don't know how to handle this yet.")
        confidence_value = confidence_value.value

        name_query = "Please suggest a title for your response."
        if self.blackboard.contributions:
            name_query += optionally_include_enumeration(
                "Make it unique. It should be different than existing sections under 'Contributions':",
                [contribution.name for contribution in self.blackboard.contributions],
            )
        name_query += (
            "Respond with the title wrapped in \"\" at the start, e.g. \"My Title\"., followed by an optional "
            "explanation using the format '## Explanation\n{your_explanation}'."
        )

        _, name_chain = chain.query(name_query)

        string_converter = StringConverter()
        name = string_converter.convert_from_chain(name_chain)
        if name.is_missing():
            raise NotImplementedError("We don't know how to handle this yet.")
        name = name.value

        # if self.blackboard.contributions:
        #     dependencies_query = "Please provide the subsection titles this solution directly depends on. "
        #     dependencies_query += optionally_include_enumeration(
        #         "(The following subsections exist under 'Contributions'",
        #         [contribution.name for contribution in self.blackboard.contributions],
        #         ")",
        #     )
        #     dependencies, _ = chain.query(dependencies_query)
        # else:
        #     dependencies = []

        return does_response, Contribution(
            name=name,
            origin="Oracle",
            content=how_response,
            feedback=feedback_response,
            confidence=confidence_value,
            confidence_full=confidence_response,
            dependencies=[],
        )


#%%
import langchain
from langchain import OpenAI
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(".chat.langchain.db")

chat_model = CachedChatOpenAI(max_tokens=512)

text_model = OpenAI(
    model_name="text-davinci-001",
    max_tokens=256,
    model_kwargs=dict(temperature=0.0),
)

# blackboard = Blackboard("Answer the question: what's the proof that the square root of 2 is irrational?")
blackboard = Blackboard(
    "Write a short essay considering whether AGI non-proliferation is achievable and how it compares to nuclear or autonomous weapon non-proliferation."
)
oracle = Oracle(chat_model, text_model)
controller = Controller(blackboard, oracle, [])

#%%
for i in range(10):
    solved = controller.update()
    print(i, solved)
    print(blackboard.to_prompt_context())
    if solved:
        break
