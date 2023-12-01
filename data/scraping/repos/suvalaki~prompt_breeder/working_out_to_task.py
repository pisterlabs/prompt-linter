from typing import Iterator
from copy import deepcopy
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from prompt_breeder.types import (
    Phenotype,
    UnitOfEvolution,
    Population,
)
from prompt_breeder.mutators.base import Mutator

# from langchain.prompts.example_selector.base import BaseExampleSelector


# Lamarkin Operator
class WorkingOutToTask(LLMChain, Mutator):
    """Fill up a few-shot context with only workings out that led to correct answers.
    During evaluation we provide this few shot-context before the task-prompt, providing
    guidance as to the form of the working out that is desired. If the few-shot context
    list is full, a single randomly sampled new correct working out replaces an existing
    working out from the list after fitness evaluation of a unit on a new set of
    questions. In addition, with a 10% chance we resample the whole context list with
    probability inverse to the maximum context list length"""

    # Get new correct workings
    correct_working_out_provider: Iterator[Phenotype]
    max_context_size: int = 2

    prompt = PromptTemplate.from_template(
        "I gave a friend an instruction and some advice. "
        "Here are the correct examples of his workings out: \n{context}\n"
        "The instruction was: "
    )

    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        unit = deepcopy(unit)
        examples = [
            next(self.correct_working_out_provider)
            for i in range(self.max_context_size)
        ]

        unit.task_prompt_set = [
            self.task_prompt_factory(
                self.run(
                    {
                        "context": "\n\n".join([str(example) for example in examples]),
                    },
                    **kwargs
                )
            )
            for member in unit.task_prompt_set
        ]
        return unit
