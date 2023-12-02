#!/usr/bin/env python
# coding: utf-8

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

import blackhc.project.script

"""LLM as CPU Spike"""
import dataclasses
import json
import re
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import langchain
import pydantic.dataclasses
from langchain import OpenAI
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM, OpenAIChat
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AIMessage, BaseOutputParser, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from blackboard_pagi.cached_chat_model import CachedChatOpenAI
from blackboard_pagi.oracle_chain import Oracle
from blackboard_pagi.prompts.chat_chain import ChatChain
from blackboard_pagi.prompts.structured_converters import (
    BooleanConverter,
    LLMBool,
    ProbabilityConverter,
    StringConverter,
)


class PydanticDataclassOutputParser(PydanticOutputParser):
    def parse(self, text: str):
        # Ignore type mismatch
        # noinspection PyTypeChecker
        return super().parse(text)


langchain.llm_cache = SQLiteCache(".execution_llm_spike.langchain.db")

# chat_model = CachedChatOpenAI(model_name="gpt-4", max_tokens=512)
chat_model = CachedChatOpenAI(max_tokens=512)

text_model = OpenAI(
    model_name="text-davinci-003",
    max_tokens=256,
    model_kwargs=dict(temperature=0.0),
)

#%%

from pydantic.dataclasses import dataclass


@dataclass
class Context:
    knowledge: dict[str, str]


# We want to define dataclasses for different actions the model can execute (e.g. "add a new contribution")
# and then use the model to decide which action to execute.
# We want to parse the actions from the model's output, and then execute them.

# Can we use pydantic discriminators to do this?

#%%
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class KnowledgeAction(BaseModel):
    """
    An action to set or remove knowledge from the context.
    """

    action: Literal["set_knowledge", "remove_knowledge"]
    key: str
    value: str | None = None

    def execute(self, context: Context):
        if self.action == "set_knowledge":
            context.knowledge[self.key] = self.value
        elif self.action == "remove_knowledge":
            del context.knowledge[self.key]
        else:
            raise ValueError(f"Unknown action {self.action}")


class FinishAction(BaseModel):
    """
    An action to signal that the goal has been reached.
    """

    action: Literal["finish"]

    def execute(self, context: Context):
        print(context)


class Action(BaseModel):
    params: KnowledgeAction | FinishAction = Field(discriminator='action')


# Test parsing from obj
action = Action.parse_obj(
    {
        "params": {
            "action": "set_knowledge",
            "key": "Goal",
            "value": "Write a short paper about blackboard pattern",
        }
    }
)
action

#%%
def processing_step(oracle: Oracle, context: Context) -> Tuple[Action, Context]:
    output_parser = PydanticOutputParser()
    output_parser.pydantic_object = Action

    chain = oracle.start_oracle_chain(
        f"---{context}\n\n---\n\nThis is the context you have access to and which you can operate on. "
        "You can add knowledge to the context, or remove knowledge from the context. "
        "You can also finish the execution of the blackboard pattern."
    )
    response, _ = chain.query("What do you want to do?\n\n---\n\n" f"{output_parser.get_format_instructions()}")

    print(response)

    action = output_parser.parse(response)

    context = deepcopy(context)
    action.params.execute(context)

    return action, context


oracle = Oracle(chat_model, text_model)
context = Context(knowledge=dict(Goal="Write a short paper about blackboard pattern"))

for _ in range(5):
    action, context = processing_step(oracle, context)
    if isinstance(action.params, FinishAction):
        break
