"""
This file creates extends the original Ask-Tell interface by incorporating contextual information for solubility
prediction.
This method adapts the prefix and the prompt template, in attempt to improve prediction accuracy.
Note that there are other ways to incorporate contextual information into the LLM.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
from typing import *
from functools import partial
from typing import Tuple, List, Any, Union

# Third Party
import numpy as np
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import FAISS, Chroma
from numpy import ndarray

# Private
from cebo.helper.distmodel import DiscreteDist, GaussDist
from cebo.models.llm import LLM
from cebo.helper.aqfxns import (
    probability_of_improvement,
    expected_improvement,
    upper_confidence_bound,
    greedy,
)

# -------------------------------------------------------------------------------------------------------------------- #
_answer_choices = ["A", "B", "C", "D", "E"]


class CEBOLIFT(LLM):
    def __init__(
        self,
        model: str,
        prompt_template: PromptTemplate = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        prefix: Optional[str] = None,
        x_formatter: Callable[[str], str] = lambda x: x,
        y_formatter: Callable[[float], str] = lambda y: f"{y:0.2f}",
        y_name: str = "output",
        x_name: str = "input",
        selector_k: Optional[int] = None,
        k: int = 5,
        verbose: bool = False,
        cos_sim: bool = False,
        features: bool = False,
        domain: str = None,
    ) -> None:
        """Initialize Ask-Tell optimizer.

        You can pass formatters that will make your data more compatible with the model. Note that
        y as output form the model must be a float(can be parsed with ``float(y_str)``)

        Args:
            prompt_template: Prompt template that should take x and y (for few shot templates)
            suffix: Matching suffix for first part of prompt template - for actual completion.
            model: OpenAI base model to use for training and inference.
            temperature: Temperature to use for inference. If None, will use model default.
            prefix: Prefix to add before all examples (e.g., some context for the model).
            x_formatter: Function to format x for prompting.
            y_formatter: Function to format y for prompting.
            y_name: Name of y variable in prompt template (e.g., density, value of function, etc.)
            x_name: Name of x variable in prompt template (e.g., input, x, etc.). Only appears in inverse prompt
            selector_k: What k to use when switching to selection mode. If None, will use all examples
            k: Number of examples to use for each prediction.
            verbose: Whether to print out debug information.
        """
        self._model = model
        self._temperature = temperature
        self._selector_k = selector_k
        self._ready = False
        self._ys = []
        self.format_x = x_formatter
        self.format_y = y_formatter
        self._y_name = y_name
        self._x_name = x_name
        self._prompt_template = prompt_template
        self._suffix = suffix
        self._prefix = prefix
        self._example_count = 0
        self._temperature = temperature
        self._k = k
        self._answer_choices = _answer_choices[:k]
        self._calibration_factor = None
        self._verbose = verbose
        self.tokens_used = 0
        self.cos_sim = cos_sim
        self.features = features
        self.domain = domain

    def set_calibration_factor(self, calibration_factor):
        self._calibration_factor = calibration_factor

    def _setup_llm(self):
        # nucleus sampling seems to get more diversity
        return self.get_llm(
            n=self._k,
            best_of=self._k,
            temperature=0.1 if self._temperature is None else self._temperature,
            model=self._model,
            top_p=1.0,
            stop=["\n", "###", "#", "##"],
            logit_bias={
                "198": -100,  # new line,
                "628": -100,  # double new line,
                "50256": -100,  # endoftext
            },
            max_tokens=256,
            logprobs=1,
        )

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        # Create input variables and template
        input_variables = list(example.keys())
        if self.features:
            template = (
                f"Q: What is the {self._y_name} of {{{input_variables[0]}}}, given the following properties: "
                + ", ".join([f"{var} is {{{var}}}" for var in input_variables[1:-1]])
                + "?"
                + f"\nA: {{{input_variables[-1]}}}###\n\n "
            )

        else:
            template = f"Q: Given {input_variables[0]}, what is {self._y_name}?\nA: {input_variables[-1]}###\n\n"
        # Setup prefix i.e. the background on the task that the LLM will perform
        if prefix is None:
            if self.domain is None:
                prefix = (
                    "The following are correctly answered questions. "
                    "Each answer is numeric and ends with ###\n"
                )
            else:
                prefix = (
                    f"You are an expert {self.domain}. "
                    "The following are correctly answered questions. "
                    "Each answer is numeric and ends with ###\n"
                    "Your task is to answer the question as accurately as possible. "
                )

        # Setup prompt template i.e. the information the LLM will process for the given problem
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=input_variables, template=template
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            elif self.features:
                suffix = (
                    f"Q: What is the {self._y_name} of {{{input_variables[0]}}} given the following properties: "
                    + ", ".join(
                        [f"{var} is {{{var}}}" for var in input_variables[1:-1]]
                    )
                    + "?"
                    + f"\nA: "
                )
            else:
                suffix = (
                    f"Q: Given {input_variables[0]}, what is the {self._y_name}?\nA: "
                )
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        # test out prompt
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        else:
            examples = []
        example_selector = None
        if self._selector_k is not None:
            # Convert list to be readable
            example = {key: str(value) for key, value in example.items()}
            if len(examples) == 0:
                raise ValueError("Cannot do zero-shot with selector")
            if not self.cos_sim:
                example_selector = (
                    example_selector
                ) = MaxMarginalRelevanceExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self._selector_k,
                )
            else:
                example_selector = (
                    example_selector
                ) = SemanticSimilarityExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    Chroma,
                    k=self._selector_k,
                )
        return FewShotPromptTemplate(
            examples=examples if example_selector is None else None,
            example_prompt=prompt_template,
            example_selector=example_selector,
            suffix=suffix,
            prefix=prefix,
            input_variables=input_variables[:-1],
        )

    def tell(self, example_dict: Dict) -> None:
        """Tell the optimizer about a new example."""
        # Add points
        self._ys.append(example_dict["Solubility"])
        # change example dictionary
        example_dict = {
            key: str(value)
            if key != "Solubility"
            else f"{value:.8f}".rstrip("0").rstrip(".")
            if value != 0
            else "0.00"
            for key, value in example_dict.items()
        }
        if not self._ready:
            self.prompt = self._setup_prompt(
                example_dict, self._prompt_template, self._suffix, self._prefix
            )
            self.llm = self._setup_llm()
            self._ready = True
        else:
            # in else, so we don't add twice
            if self._selector_k is not None:
                self.prompt.example_selector.add_example(example_dict)
            else:
                self.prompt.examples.append(example_dict)
        self._example_count += 1

    def predict(self, x: Dict) -> Union[tuple[Any, list[str]], Any]:
        """Predict the probability distribution and values for a given x.

        Args:
            x: The x value(s) to predict.
        Returns:
            The probability distribution and values for the given x.

        """
        if not self._ready:
            # special zero-shot
            self.prompt = self._setup_prompt(
                None, self._prompt_template, self._suffix, self._prefix
            )
            self.llm = self._setup_llm()
            self._ready = True
        if self._selector_k is not None:
            # have to update this until my PR is merged
            self.prompt.example_selector.k = min(self._example_count, 10)
        if not isinstance(x, list):
            x = {key: str(value) for key, value in x.items()}
            queries = [self.prompt.format(**x)]
        else:
            queries = [
                self.prompt.format(**{key: str(value) for key, value in x_i.items()})
                for x_i in x
            ]
        results, tokens = self._predict(queries)
        self.tokens_used += tokens
        # need to replace any GaussDist with pop std
        for i, result in enumerate(results):
            if len(self._ys) > 1:
                ystd = np.std(self._ys)
            elif len(self._ys) == 1:
                ystd = self._ys[0]
            else:
                ystd = 10
            if isinstance(result, GaussDist):
                results[i].set_std(ystd)
        if self._calibration_factor:
            for i, result in enumerate(results):
                if isinstance(result, GaussDist):
                    results[i].set_std(result.std() * self._calibration_factor)
                elif isinstance(result, DiscreteDist):
                    results[i] = GaussDist(
                        results[i].mean(),
                        results[i].std() * self._calibration_factor,
                    )
        # Compute mean and standard deviation
        if len(results) > 1:
            return results, queries
        else:
            return results[0], queries

    def ask(
        self,
        data,
        possible_x: List[str],
        _lambda: float = 0.5,
    ) -> Dict:
        """Ask the optimizer for the next x to try.
        Args:
            possible_x: List of possible x values to choose from.
            _lambda: Lambda value to use for UCB.
        Return:
            The selected x values, their acquisition function values, and the predicted y modes.
            Sorted by acquisition function value (descending)
        """
        # Store highest value so far
        if len(self._ys) == 0:
            best = 0
        else:
            best = np.max(self._ys)
        # Create list of values to query over
        possible_x_l = list(possible_x)
        # Calculate results over 3 acquisition functions
        aq_fxns = {
            "Expected Improvement": expected_improvement,
            "Upper Confidence Bound": partial(upper_confidence_bound, _lambda=_lambda),
        }
        # Obtain results for each acquisition function value
        results = self._ask(data, possible_x_l, best, aq_fxns)
        # If we have nothing then just go random
        return results

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        # implementation of tell
        if alt_ys is not None:
            if len(alt_ys) != len(self._answer_choices) - 1:
                raise ValueError("Must provide 4 alternative ys.")
            alt_ys = [self.format_y(alt_y) for alt_y in alt_ys]
        else:
            alt_ys = []
            alt_y = y
            for i in range(100):
                if len(alt_ys) == len(self._answer_choices) - 1:
                    break
                if i < 50:
                    alt_y = y * 10 ** np.random.normal(0, 0.2)
                else:  # try something different
                    alt_y = y + np.random.uniform(-10, 10)
                if self.format_y(alt_y) not in alt_ys and self.format_y(
                    alt_y
                ) != self.format_y(y):
                    alt_ys.append(self.format_y(alt_y))
        # choose answer
        answer = np.random.choice(self._answer_choices)
        example_dict = dict(
            x=self.format_x(x),
            Answer=answer,
            y_name=self._y_name,
        )
        for a in self._answer_choices:
            if a == answer:
                example_dict[a] = self.format_y(y)
            else:
                example_dict[a] = alt_ys.pop()
        self._ys.append(y)
        inv_example = dict(
            x=self.format_x(x),
            y_name=self._y_name,
            y=self.format_y(y),
            x_name=self._x_name,
        )
        return example_dict, inv_example

    def _predict(self, queries: List[str]) -> tuple[Any, Any]:
        result, token_usage = self.openai_topk_predict(queries, self.llm, self._verbose)
        return result, token_usage

    def _ask(
        self, data, possible_x: List[str], best: float, aq_fxns: Dict[str, Callable]
    ) -> Dict:
        # Obtain results and queries
        results, queries = self.predict(possible_x)
        # Calculate acquisition function value
        final_results = {}
        for aq_fxn_name, aq_fxn in aq_fxns.items():
            aq_vals = np.array(
                [aq_fxn(r, best) if len(r) > 0 else np.nan for r in results]
            )
            if aq_fxn_name == "Upper Confidence Bound":
                # Check UCB range
                target_vals = [
                    data[
                        (data["SMILES"] == example["SMILES"])
                        & (data["SMILES Solvent"] == example["SMILES Solvent"])
                    ]["Solubility"].values[0]
                    for example in possible_x
                ]
                num_success_bounds = sum(
                    [
                        1 if result_range[0] <= target_val <= result_range[1] else 0
                        for result_range, target_val in zip(aq_vals, target_vals)
                    ]
                ) / len(possible_x)
                # Final acquisition values
                aq_vals = aq_vals[:, 1]
                # Other acquisition values
                aq_vals_cleaned = np.where(
                    np.isnan(aq_vals),
                    -np.inf,
                    np.where(np.isinf(aq_vals), 1e10, aq_vals),
                )
                selected = np.argmax(aq_vals_cleaned)
                final_results[f"{aq_fxn_name}"] = {
                    "Selected": possible_x[selected],
                    "Acquisition Values": aq_vals_cleaned,
                    "Number of points contained in acquisition range": num_success_bounds,
                }
            if aq_fxn_name == "Expected Improvement":
                # Other acquisition values
                aq_vals_cleaned = np.where(
                    np.isnan(aq_vals),
                    -np.inf,
                    np.where(np.isinf(aq_vals), 1e10, aq_vals),
                )
                selected = np.argmax(aq_vals_cleaned)
                final_results[f"{aq_fxn_name}"] = {
                    "Selected": possible_x[selected],
                    "Acquisition Values": aq_vals_cleaned,
                    "Number of points contained in acquisition range": "N/A",
                }
        # Add random
        final_results["random"] = {
            "Selected": np.random.choice(possible_x),
            "Acquisition Values": [0],
            "Number of points contained in acquisition range": None,
        }
        return final_results
