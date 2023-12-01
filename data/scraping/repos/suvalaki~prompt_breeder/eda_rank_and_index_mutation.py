from typing import List, Callable, Type
from langchain.llms.base import BaseLanguageModel
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance,
    EmbeddingDistanceEvalChain,
)
from prompt_breeder.types import TaskPrompt, MutationPrompt
from prompt_breeder.evolution.fitness import FitnessScorer
from prompt_breeder.mutators.estimation_of_distribution_mutation import (
    EstimationOfDistributionMutation,
)


class EdaRankAndIndexMutation(EstimationOfDistributionMutation):
    fitness_scorer: FitnessScorer
    prompt = PromptTemplate.from_template(
        "INSTRUCTION: {mutation_prompt}"
        + "\n A List ofResponses in descending order of score. "
        + "{task_prompt_set}  INSTRUCTION MUTATNT: "
    )

    @classmethod
    def from_llm(  # type: ignore[override]
        cls: Type["EdaRankAndIndexMutation"],
        task_prompt_factory: Callable[[str], TaskPrompt],
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        distance_metric: EmbeddingDistance,
        fitness_scorer: FitnessScorer,
        **kwargs,
    ) -> "EdaRankAndIndexMutation":
        return cls(
            task_prompt_factory=task_prompt_factory,
            mutation_prompt_factory=mutation_prompt_factory,
            llm=llm,
            embed_scorer=EmbeddingDistanceEvalChain(
                embeddings=embeddings, distance_metric=distance_metric
            ),
            fitness_scorer=fitness_scorer,
            **kwargs,
        )

    def sort_population(
        self, task_prompt_set: List[TaskPrompt], run_manager=None, **kwargs
    ) -> None:
        # sort by fitness
        cb = run_manager.get_child() if run_manager else None
        fitnesses = [
            self.fitness_scorer.score(prompt, callbacks=cb)
            for prompt in task_prompt_set
        ]
        pairs = list(zip(task_prompt_set, fitnesses))
        pairs.sort(key=lambda x: x[1], reverse=True)
        task_prompt_set[:] = [pair[0] for pair in pairs]
