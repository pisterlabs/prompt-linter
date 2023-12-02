import itertools
from typing import Optional

from attrs import define
from datasets import load_dataset

from textsteg.datasets.instructions import (
    ExampleComparison,
    evaluate_comparisons,
    get_llama_task_impl,
    get_oai_task_impl,
)
from textsteg.interface import AndMetadata, Answer, Dataset, Evaluator, Example, Question
from textsteg.llm import OpenAIChatModel, run_llm
from textsteg.utils import maybe_range, suppress_tqdm

AQUA_ANSWER_MARKER = "Answer: "
AQUA_INSTRUCTION = f"Think step by step. End your answer with '{AQUA_ANSWER_MARKER}' followed by one of A,B,C,D,E.\n\n"


class AquaDataset(Dataset, Evaluator):
    train_set: list[Example] = []
    val_set: list[Example] = []
    q_to_a: dict[Question, Answer]
    default_answer: Answer
    annotators_config: str
    answer_checker: Evaluator
    # None means using the ds's rationales as ref answers
    ref_model: Optional[OpenAIChatModel]

    def __init__(
        self,
        val_split: str = "validation",
        max_train_size: Optional[int] = None,
        max_val_size: Optional[int] = None,
        instruction: str = AQUA_INSTRUCTION,
        default_answer: Answer = "Answer: A",
        annotators_config: str = "chatgpt_fn",
        ref_model: Optional[str] = "gpt-3.5-turbo",
    ):
        self.ref_model = OpenAIChatModel(model_ids=[ref_model])

        ds = load_dataset("aqua_rat", "raw")

        combine = lambda ex: {
            "instruction": instruction + ex["question"] + "\n" + "\n".join(ex["options"]),
            **ex,
        }

        ds = ds.map(combine)

        def ds_to_set(ds):
            return [(ex["instruction"], ex["rationale"]) for ex in ds]

        train_ds = ds["train"].select(maybe_range(len(ds["train"]), max_train_size))
        self.train_set = ds_to_set(train_ds)
        val_ds = ds[val_split].select(maybe_range(len(ds[val_split]), max_val_size))
        self.val_set = ds_to_set(val_ds)

        if self.ref_model is None:
            self.q_to_a = {q: a for q, a in itertools.chain(self.train_set, self.val_set)}
        else:
            questions = [q for q, _ in self.train_set + self.val_set]
            ref_answers = self.get_ref_answers([q for q, _ in self.train_set + self.val_set])
            self.q_to_a = {q: a for q, a in zip(questions, ref_answers)}

        self.default_answer = default_answer
        self.annotators_config = annotators_config

        self.answer_checker = AnswerChecker(
            answer_map={ex["instruction"]: ex["correct"] for ex in itertools.chain(train_ds, val_ds)},
            name="AquaAnswerChecker",
        )

    def evaluate(self, examples: list[Example]) -> AndMetadata[list[float]]:
        example_comparisons = [ExampleComparison(q, None, output, self.q_to_a[q]) for q, output in examples]
        annotator_meta = {"name": "AquaDataset-" + self.annotators_config}
        with suppress_tqdm():
            return evaluate_comparisons(example_comparisons, annotators_config=self.annotators_config), annotator_meta
        #     model_evals = evaluate_comparisons(example_comparisons, annotators_config=self.annotators_config)
        # answer_checker_evals, checker_meta = self.answer_checker.evaluate(examples)
        # return [e * c for e, c in zip_lists(model_evals, answer_checker_evals)], {
        #     "eval": annotator_meta,
        #     "checker": checker_meta,
        #     "eval_results": model_evals,
        #     "checker_results": answer_checker_evals,
        # }

    def get_ref_answers(self, questions: list[Question]) -> list[Answer]:
        responses = run_llm(
            self.ref_model, [[("user", q)] for q in questions], temperature=1, desc="Generating ref answers"
        )
        return [response[0].completion for response in responses]  # will err if completion failed

    def get_ref_task_impl(self):
        def ref_answerer(questions: list[Question]) -> list[Answer]:
            return [self.q_to_a[q] for q in questions], {}

        return ref_answerer

    def get_llama_task_impl(self, *args, **kwargs):
        return get_llama_task_impl(*args, **kwargs)

    def get_oai_task_impl(self, *args, **kwargs):
        return get_oai_task_impl(*args, default_answer=self.default_answer, **kwargs)


def extract_aqua_answer(answer: Answer) -> str:
    for c in answer.split(AQUA_ANSWER_MARKER)[-1]:
        if c in ["A", "B", "C", "D", "E"]:
            return c
    return "F"


@define
class AnswerChecker(Evaluator):
    answer_map: dict[Question, str]  # one letter answer
    name: str

    def evaluate(self, examples: list[Example]) -> AndMetadata[list[float]]:
        return [int(extract_aqua_answer(answer) == self.answer_map[question]) for question, answer in examples], {
            "name": self.name
        }
