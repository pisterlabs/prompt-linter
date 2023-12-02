from typing import Callable, Iterator
from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from prompt_breeder.types import TaskPrompt, MutationPrompt, ThinkingStyle
from prompt_breeder.mutators.base import Mutator, Hypermutation
from prompt_breeder.mutators.first_order_prompt_generation import (
    FirstOrderPromptGeneration,
)


class ZeroOrderMutationInitialization(LLMChain, Mutator):
    prompt = PromptTemplate.from_template("{problem_description} {thinking_style}")


class ZeroOrderHypermutation(Hypermutation):
    """We concatenate the original problem description to a randomly sam-
    pled thinking-style, and feed it to the LLM to generate a new mutation-prompt. The
    resulting mutation-prompt is applied to a task-prompt to make a variant of the
    task-prompt as in First-order Prompt Generation (see Section 3.2.1). Note that this
    zero-order meta-mutation operator is identical to that used during initialization.

    We generate the initial task-prompts by concatenating (for each task-
    prompt) a randomly drawn ‘mutation-prompt’ (e.g. "Make a variant of the prompt.")
    and a randomly drawn ‘thinking-style’ (e.g. "Let’s think step by step") to the
    problem description, and provide that to the LLM to produce a continuation,
    resulting in an initial task-prompt
    """

    mutate_mutator_chain: ZeroOrderMutationInitialization
    mutate_task_prompt_chain: FirstOrderPromptGeneration

    @classmethod
    def from_llm(
        cls,
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        task_prompt_factory: Callable[[str], TaskPrompt],
        thinking_style_provider: Iterator[ThinkingStyle],
        llm: BaseLanguageModel,
        **kwargs
    ):
        return cls(
            task_prompt_factory=task_prompt_factory,
            mutation_prompt_factory=mutation_prompt_factory,
            thinking_style_provider=thinking_style_provider,
            mutate_mutator_chain=ZeroOrderMutationInitialization(
                llm=llm,
                task_prompt_factory=task_prompt_factory,
                mutation_prompt_factory=mutation_prompt_factory,
                **kwargs,
            ),
            mutate_task_prompt_chain=FirstOrderPromptGeneration(
                llm=llm,
                task_prompt_factory=task_prompt_factory,
                mutation_prompt_factory=mutation_prompt_factory,
                **kwargs,
            ),
            **kwargs,
        )
