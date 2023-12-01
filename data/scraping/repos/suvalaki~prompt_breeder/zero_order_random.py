from typing import Iterator, Dict, Callable
from langchain.llms.base import BaseLanguageModel
from prompt_breeder.types import (
    ProblemDescription,
    TaskPrompt,
    MutationPrompt,
    ThinkingStyle,
    UnitOfEvolution,
    Population,
)
from prompt_breeder.evolution.initialization.base import UnitInitialization
from prompt_breeder.mutators.zero_order_hypermutation import ZeroOrderHypermutation


class ZeroOrderInitialization(UnitInitialization):
    """Draw a random thinking style and mutation prompt and ZeroOrderHypermutation"""

    mutation_prompt_provider: Iterator[MutationPrompt]
    mutator: ZeroOrderHypermutation

    def _call(
        self, inputs: Dict[str, str], run_manager=None, **kwargs
    ) -> Dict[str, UnitOfEvolution]:
        cb = run_manager.get_child() if run_manager else None
        initial_unit = UnitOfEvolution(
            problem_description=self.problem_description_factory(
                inputs["problem_description"]
            ),
            task_prompt_set=self.n_members_per_unit
            * [self.mutator.task_prompt_factory(str(inputs["problem_description"]))],
            mutation_prompt=next(self.mutation_prompt_provider),
        )
        final_unit = self.mutator.mutate(
            Population(members=[]), initial_unit, callbacks=cb
        )
        return {self.output_key: final_unit}

    @classmethod
    def from_llm(
        cls,
        problem_description_factory: Callable[[str], ProblemDescription],
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        task_prompt_factory: Callable[[str], TaskPrompt],
        thinking_style_provider: Iterator[ThinkingStyle],
        mutation_prompt_provider: Iterator[MutationPrompt],
        llm: BaseLanguageModel,
        n_members_per_unit: int,
        **kwargs
    ):
        return cls(
            problem_description_factory=problem_description_factory,
            n_members_per_unit=n_members_per_unit,
            mutation_prompt_provider=mutation_prompt_provider,
            mutator=ZeroOrderHypermutation.from_llm(
                mutation_prompt_factory,
                task_prompt_factory,
                thinking_style_provider,
                llm,
                **kwargs
            ),
            **kwargs
        )
