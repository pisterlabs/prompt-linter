import os
from abc import ABC
from typing import List, Any, Dict

import numpy as np
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from src.models.meta_models import MetaModelForSequenceClassification, OpenAIModelInfo
from src.model_strategies import ProbabilityBasedStrategy
import json

from src.model_strategies.abstract import ModelStrategy


class AdaptiveOpenAI(ABC):
    strategy: ModelStrategy
    generation_parameters: Dict[str, Dict]

    def __init__(
            self,
            strategy: ModelStrategy,
            generation_parameters: Dict[str, Dict],
            api_key: str = None
    ):
        self.strategy = strategy
        self.generation_parameters = generation_parameters
        self.api_key = api_key

    def _format_with_demonstrations(self, batch: List[str], fs_prompt: FewShotPromptTemplate):
        input_key = fs_prompt.input_variables[0]
        return [fs_prompt.format(**{input_key: input_data}) for input_data in batch]

    @staticmethod
    def __call(
            batch: List,
            model_name: str,
            api_key: str,
            generation_parameters: Dict[str, Any]
    ):
        backend = OpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            verbose=True,
            **generation_parameters,
        )

        return backend.generate(batch)

    def __call__(
            self,
            batch: List[str],
            api_key: str = None,
            few_shot_prompts: Dict[str, FewShotPromptTemplate] = None,
            generation_parameters: Dict[str, Dict] = None,
            return_decisions: bool = False
    ):
        model_assignment, assigned_cost = self.strategy(batch) # TODO calculate cost based on the whole prompt

        if generation_parameters is None:
            generation_parameters = self.generation_parameters

        if api_key is None:
            api_key = self.api_key

        answers = []
        unique_models = np.unique(model_assignment)
        for unique_model in unique_models:
            curr_batch = [data for model, data in zip(model_assignment, batch) if model == unique_model]

            if few_shot_prompts:
                fs_prompt = few_shot_prompts[unique_model]
                curr_batch = self._format_with_demonstrations(curr_batch, fs_prompt)

            answers.extend(
                self.__call(
                    batch=curr_batch,
                    model_name=unique_model,
                    api_key=api_key,
                    generation_parameters=generation_parameters[unique_model]
                )
            )

        if return_decisions:
            return {"answers": answers, "selected_models": model_assignment, "estimated_costs": assigned_cost}
        else:
            return answers
