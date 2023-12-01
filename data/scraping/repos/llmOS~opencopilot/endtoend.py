from typing import List
from opencopilot import settings

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain


from opencopilot.eval.entities import (
    EndToEndExample,
    EndToEndResult,
    EndToEndDataset,
    EndToEndSingleEvaluation,
    EndToEndSummaryEvaluation,
)

PROMPT_TEMPLATE = """You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to grade the student answer as either A, B, C, D, or F.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: A, B, C, D or F here

Grade the student answers based ONLY on their factual accuracy, helpfulness and completeness. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""
PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=PROMPT_TEMPLATE
)


def get_summary_evaluation(
    evaluations: List[EndToEndSingleEvaluation],
) -> EndToEndSummaryEvaluation:
    """Roll up a list of single evaluations into a summary evaluation."""
    evaluations_count = len(evaluations)
    total_score = 0
    rubric = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "F": 0.0}
    for evaluation in evaluations:
        grade = rubric.get(evaluation.evaluation)
        if grade:  # TODO - failures of grading are implicitly scored as zero
            total_score += grade
    return EndToEndSummaryEvaluation(
        evaluations_count=evaluations_count,
        evaluations_score=round((total_score / evaluations_count) * 100, 4),
        single_evaluations=evaluations,
    )


def evaluate_endtoend_single(
    example: EndToEndExample, prediction: EndToEndResult
) -> EndToEndSingleEvaluation:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    # patch it for context token limit - TODO is this still needed?
    settings.get().MODEL = "gpt-4"

    eval_chain = QAEvalChain.from_llm(llm=llm, prompt=PROMPT)
    eval_example = {"query": example.query, "answer": example.answer}
    eval_prediction = {"result": prediction.answer}

    graded_output = eval_chain.evaluate([eval_example], [eval_prediction])[0]
    print("graded_output:", graded_output)

    return EndToEndSingleEvaluation(evaluation=graded_output["results"])


def evaluate_endtoend_dataset(
    dataset: EndToEndDataset, predictions: List[EndToEndResult]
) -> EndToEndSummaryEvaluation:
    assert len(dataset.examples) == len(
        predictions
    ), "Dataset and predictions must be the same length"

    # Convert the results into our own format
    evaluations = [
        evaluate_endtoend_single(example, prediction)
        for example, prediction in zip(dataset.examples, predictions)
    ]

    summary_evaluation = get_summary_evaluation(evaluations)
    return summary_evaluation
