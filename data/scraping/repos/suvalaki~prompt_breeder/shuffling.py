from typing import Iterator, Dict, List
from random import random, randint
from copy import deepcopy

from prompt_breeder.types import (
    Phenotype,
    FewShowUnitOfEvolution,
    Population,
)
from prompt_breeder.mutators.base import Mutator

# from langchain.prompts.example_selector.base import BaseExampleSelector


# This is to occur after fitness evaluation
class ContextShuffling(Mutator):
    """Fill up a few-shot context with only workings out that led to correct answers.
    During evaluation we provide this few shot-context before the task-prompt, providing
    guidance as to the form of the working out that is desired. If the few-shot context
    list is full, a single randomly sampled new correct working out replaces an existing
    working out from the list after fitness evaluation of a unit on a new set of
    questions. In addition, with a 10% chance we resample the whole context list with
    probability inverse to the maximum context list length"""

    # Get new correct workings
    correct_working_out_provider: Iterator[Phenotype]
    max_context_size: int = 10
    probability_of_refresh_full_list: float = 0.1

    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["unit"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def fill_unit_context(
        self, unit: FewShowUnitOfEvolution, **kwargs
    ) -> FewShowUnitOfEvolution:
        unit = deepcopy(unit)
        unit.contexts[:] = [
            next(self.correct_working_out_provider)
            for i in range(self.max_context_size)
        ]
        return unit

    def replace_single_context(
        self, unit: FewShowUnitOfEvolution, **kwargs
    ) -> FewShowUnitOfEvolution:
        unit = deepcopy(unit)
        idx = randint(0, len(unit.contexts) - 1)
        unit.contexts[idx] = next(self.correct_working_out_provider)
        return unit

    def mutate(  # type: ignore[override]
        self, population: Population, unit: FewShowUnitOfEvolution, **kwargs
    ) -> FewShowUnitOfEvolution:
        return self.run({"unit": unit}, **kwargs)

    def _call(
        self, inputs: Dict[str, FewShowUnitOfEvolution], run_manager=None, **kwargs
    ) -> Dict[str, FewShowUnitOfEvolution]:
        unit = inputs["unit"]
        # Fill up the context which is currently empy
        if len(unit.contexts) < self.max_context_size:
            return {self.output_key: self.fill_unit_context(unit)}

        # resample the whole context list with probability inverse to the
        # maximum context list length I leave this as a stable hyperparam instead.
        if len(unit.contexts) >= self.max_context_size:
            if random() < self.probability_of_refresh_full_list:
                return {self.output_key: self.fill_unit_context(unit)}

        # if the few-shot context list is full, a single randomly sampled
        # new correct working out replaces an existing working
        if len(unit.contexts) == self.max_context_size:
            return {self.output_key: self.replace_single_context(unit)}

        return {self.output_key: deepcopy(unit)}
