from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset
from langchain.chains.base import Chain
from langchain.evaluation import EvaluatorType, load_evaluator

from master_thesis.base import BaseMetric, BasePrompt
from master_thesis.datasets import SQUADv1Dataset
from master_thesis.defaults import RANDOM_SEED
from master_thesis.experiments import Experiment


# Prompts
class AdvancedLLaMAPrompt(BasePrompt):
    def run(self, references: Optional[List[str]], question: str) -> str:
        prompt = f"""<s>[INST] <<SYS>>Answer the question based on the context below:
        * Only provide the answer, no extra explanation.
        * Only use words from the context below in the answer.
        * If the question cannot be answered using the context or if the context is missing, answer with '-'.
        
        -----
        {self._format_references(references)}
        <</SYS>>

        Q: {question} [/INST]
        A: 
        """

        return self._model.generate(self._format_prompt(prompt))


# Datasets
class NoContextDataset(SQUADv1Dataset):
    _percent_no_context: float

    def __init__(
        self,
        percent_no_context: int = 0.10,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> None:
        super().__init__(train_limit=train_limit, test_limit=test_limit)

        self._percent_no_context = percent_no_context

    @property
    def train(self) -> Dataset:
        if not self._train:
            self._train = (
                self._dataset_dict["train"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["train"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            # Add no context samples
            self._train = self._train.map(
                self._introduce_no_context_samples(
                    0,
                    self._train.num_rows * self._percent_no_context,
                ),
                batched=False,
                with_indices=True,
            ).shuffle(seed=RANDOM_SEED)

            if self._train_limit:
                self._train = self._train.select(range(self._train_limit))

        return self._train

    @property
    def test(self) -> Dataset:
        if not self._test:
            self._test = (
                self._dataset_dict["validation"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["validation"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            # Add no context samples
            self._test = self._test.map(
                self._introduce_no_context_samples(
                    0,
                    self._test.num_rows * self._percent_no_context,
                ),
                batched=False,
                with_indices=True,
            ).shuffle(seed=RANDOM_SEED)

            if self._test_limit:
                self._test = self._test.select(range(self._test_limit))

        return self._test

    def _introduce_no_context_samples(
        self,
        start_idx: int,
        end_idx: int,
    ) -> Callable[[Dict[str, Any], int], Dict[str, Any]]:
        def map_function(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
            if idx >= start_idx and idx <= end_idx:
                return {**sample, "references": [""], "answer": "-"}
            return sample

        return map_function


# Metrics
class LevenshteinSimilarityMetric(BaseMetric):
    _evaluator: Chain

    def __init__(self) -> None:
        super().__init__()

        self._evaluator = load_evaluator(EvaluatorType.STRING_DISTANCE)

    def test(self, question: str, answer: str, prediction: str) -> float:
        return (
            1
            - self._evaluator.evaluate_strings(
                prediction=prediction,
                reference=answer,
            )["score"]
        )


# Experiment
if __name__ == "__main__":
    advanced_prompt_experiment = Experiment(
        experiment_name="case_study_advanced_prompt",
        model="meta-llama/Llama-2-7b-chat-hf",
        model_config={
            "max_tokens": 1024,
            "stop_sequences": ["\n"],
            "temperature": 0.7,
            "top_p": 1.0,
        },
        dataset=NoContextDataset,
        dataset_config={"test_limit": 1000},
        metric=LevenshteinSimilarityMetric,
        metric_config={},
        prompt=AdvancedLLaMAPrompt,
        prompt_config={},
    )
    advanced_prompt_experiment.run()
