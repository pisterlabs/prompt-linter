from typing import List, Dict, Optional


from datasets import load_dataset
import evaluate
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage


from .base import BaseTask
from ...utils.models import get_chat_model


class SLF5KTask(BaseTask):
    """Example task"""

    def get_dataset(self, phase: str, ids: Optional[List[int]] = None):
        dataset = load_dataset("JeremyAlain/SLF5K")[phase]
        updated_dataset = []
        if ids is not None:
            for element in dataset:
                if element["id"] in ids:
                    updated_dataset.append(element)
        return dataset if updated_dataset == [] else updated_dataset

    def get_chain(
        self,
        generation_llm: str,
        feedback_llm: str,
        refinement_llm: str,
        chain_name: Optional[str] = None,
    ):
        # 0. Setup
        assert chain_name in ["whole_model", "human_feedback", "model_feedback"]
        initial_llm = get_chat_model(model_name=generation_llm, max_tokens=60)
        feedback_llm = get_chat_model(model_name=feedback_llm)
        refinement_llm = get_chat_model(model_name=refinement_llm, max_tokens=60)

        if chain_name == "whole_model":
            # === 1. Initial solution === #
            initial_solution_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are an assistant for generating summaries."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
    Title: {title}
    Text: {post}
    TL;DR:
            """.strip(),
                        input_variables=["title", "post"],
                    ),
                ]
            )
            initial_solution_chain = LLMChain(
                llm=initial_llm,
                prompt=initial_solution_prompt,
                output_key="initial_solution",
            )

            # === 2. Feedback === #
            feedback_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are an assistant for generating summaries."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
    The following is a proposed summary.

    Title: {title}
    Text: {post}
    TL;DR: {initial_solution}

    Please provide feedback on the proposed solution. 
    Feedback: 
            """.strip(),
                        input_variables=["title", "post", "initial_solution"],
                    ),
                ]
            )
            feedback_chain = LLMChain(
                llm=feedback_llm, prompt=feedback_prompt, output_key="feedback"
            )

            # === 3. Refinement === #
            # Simulate an exchange
            refinement_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are an assistant for generating summaries."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
Title: {title}
Text: {post}
TL;DR:
                """.strip(),
                        input_variables=["title, post"],
                    ),
                    AIMessagePromptTemplate.from_template(
                        """
{initial_solution}
                """.strip(),
                        input_variables=["initial_solution"],
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
I'm not sure about that. Rewrite the summary making sure that incorporates the following feedback: {feedback}
                """.strip(),
                        input_variables=["feedback"],
                    ),
                ]
            )
            refinement_chain = LLMChain(
                llm=refinement_llm, prompt=refinement_prompt, output_key="refinement"
            )

            ilf_chain = SequentialChain(
                chains=[initial_solution_chain, feedback_chain, refinement_chain],
                input_variables=["title", "post"],
                output_variables=["initial_solution", "feedback", "refinement"],
            )

        elif chain_name == "human_feedback":
            # === 3. Refinement === #
            # Simulate an exchange
            refinement_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are an assistant for generating summaries."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
Title: {title}
Text: {post}
TL;DR:
                """.strip(),
                        input_variables=["title, post"],
                    ),
                    AIMessagePromptTemplate.from_template(
                        """
{generated_summary_for_feedback}
                """.strip(),
                        input_variables=["generated_summary_for_feedback"],
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
I'm not sure about that. Rewrite the summary making sure that incorporates the following feedback: {feedback}
                """.strip(),
                        input_variables=["feedback"],
                    ),
                ]
            )
            refinement_chain = LLMChain(
                llm=refinement_llm, prompt=refinement_prompt, output_key="refinement"
            )

            ilf_chain = SequentialChain(
                chains=[refinement_chain],
                input_variables=[
                    "title",
                    "post",
                    "generated_summary_for_feedback",
                    "feedback",
                ],
                output_variables=[
                    "refinement",
                ],
            )

        elif chain_name == "model_feedback":
            # === 2. Feedback === #
            feedback_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are an assistant for generating summaries."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
    The following is a proposed summary.

    Title: {title}
    Text: {post}
    TL;DR: {generated_summary_for_feedback}

    Please provide feedback on the proposed solution.
    Feedback: 
            """.strip(),
                        input_variables=[
                            "title",
                            "post",
                            "generated_summary_for_feedback",
                        ],
                    ),
                ]
            )
            feedback_chain = LLMChain(
                llm=feedback_llm, prompt=feedback_prompt, output_key="feedback"
            )

            # === 3. Refinement === #
            # Simulate an exchange
            refinement_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are an assistant for generating summaries."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
Title: {title}
Text: {post}
TL;DR:
                """.strip(),
                        input_variables=["title, post"],
                    ),
                    AIMessagePromptTemplate.from_template(
                        """
{generated_summary_for_feedback}
                """.strip(),
                        input_variables=["generated_summary_for_feedback"],
                    ),
                    HumanMessagePromptTemplate.from_template(
                        """
I'm not sure about that. Rewrite the summary making sure that incorporates the following feedback: {feedback}
                """.strip(),
                        input_variables=["feedback"],
                    ),
                ]
            )
            refinement_chain = LLMChain(
                llm=refinement_llm, prompt=refinement_prompt, output_key="refinement"
            )

            ilf_chain = SequentialChain(
                chains=[feedback_chain, refinement_chain],
                input_variables=["title", "post", "generated_summary_for_feedback"],
                output_variables=[
                    "feedback",
                    "refinement",
                ],
            )
        else:
            raise KeyError(chain_name)
        return ilf_chain

    def evaluate(self, phase: str, outputs: List[Dict]):
        # Rouge evaluation by now
        all_ids = [elem["id"] for elem in outputs]
        dataset = self.get_dataset(phase=phase)
        metric = evaluate.load("rouge")
        gold_feedback, gold_refinement = [], []
        model_initial_summary, model_refinement, model_feedback = [], [], []
        for row, example in zip(outputs, dataset):
            gold_feedback.append(example["feedback"])
            model_feedback.append(row["feedback"])
            gold_refinement.append(example["ideal_human_summary"])
            # 48 tokens max_length
            model_refinement.append(
                ".".join(" ".join(row["refinement"].split()[:48]).split(".")[:-1]) + "."
            )
            try:
                # 48 tokens max_length
                model_initial_summary.append(
                    ".".join(
                        " ".join(row["initial_solution"].split()[:48]).split(".")[:-1]
                    )
                    + "."
                )
            except KeyError:
                model_initial_summary.append(example["generated_summary_for_feedback"])
        results = {
            "feedback_rouge": metric.compute(
                predictions=model_feedback, references=gold_feedback
            )["rouge1"],
            "initial_rouge": metric.compute(
                predictions=model_initial_summary, references=gold_refinement
            )["rouge1"],
            "refinement_rouge": metric.compute(
                predictions=model_refinement, references=gold_refinement
            )["rouge1"],
        }
        return results