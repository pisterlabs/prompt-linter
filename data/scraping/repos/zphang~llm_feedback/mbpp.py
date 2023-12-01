from typing import List, Dict, Optional
import datasets

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

from .base import BaseTask
from ...utils.models import get_chat_model


class MBPPTask(BaseTask):
    """Example task"""

    def get_dataset(self, phase: str):
        return datasets.load_dataset("mbpp", split=phase)

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = "regular"):
        # 0. Setup
        initial_llm = get_chat_model(model_name=generation_llm)
        feedback_llm = get_chat_model(model_name=feedback_llm)
        refinement_llm = get_chat_model(model_name=refinement_llm)

        initial_solution_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful Python coding assistant."),
            HumanMessagePromptTemplate.from_template("""
You will be given a Python programming task and one unit test. Write a function that satisfies the specification in task description and passes the unit test. Imporant: Do not include the test case in your solution! Output just the improved solution, without any additional comments. Your entire output should be ready to be copy-pasted into a Python console and run.
Instruction:
{text}
Unit test:
{test_list_0}
Solution:
            """.strip(), input_variables=["text", "test_list_0"]),
        ])

        if chain_name == "regular":
            feedback_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful Python coding assistant."),
                HumanMessagePromptTemplate.from_template("""
You will be given a Python programming task, unit tests and a candidate solution. Your job is to provide short feedback on how to improve the candidate solution such that it satisfies the specification in task description and passes the unit test. Be as concise as possible! Do not provide the corrected solution, limit yourself to short feedback in natural language. Focus on correctness, not on following Python style guide or good variable naming. Don't comment on the provided unit tests, they're fixed and not meant to be changed. Your feedback should be understandable to someone who doesn't see these unit tests. If the solution is already okay, just output \"OK\".
Instruction:
{text}

Unit tests:
{test_list_0}
{test_list_1}
{test_list_2}

Solution:
{initial_solution}
            """.strip(), input_variables=["text", "test_list_0", "test_list_1", "test_list_2" "initial_solution"]),
            ])
            refinement_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful Python coding assistant."),
                HumanMessagePromptTemplate.from_template("""
Feedback:
You will be given a Python programming task, one unit test, an initial solution and feedback an expert provided on that initial solution. Your job is to rewrite the initial solution based on the feedback. Output just the improved solution, without any additional comments. Don't include unit test in your improved solution, they are not part of the solution. Your entire output should be ready to be copy-pasted into a Python console and run.

Instruction:
{text}

Unit test:
{test_list[0]}

Initial solution:
{initial_solution}

Feedback:
{feedback}
Improved solution:
            """.strip(), input_variables=[
                    "text", "test_list_0", "test_list_1", "test_list_2", "initial_solution", "feedback"
                ]),
            ])
        elif chain_name == "chat":
            feedback_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful Python coding assistant. A human will show you a Python programming task, unit tests for this task and a candidate solution that human wrote. Your job is to provide short feedback on how to improve human's candidate solution such that it satisfies the specification in task description and passes the unit test. Be as concise as possible! Do not provide the corrected solution, limit yourself to short feedback in natural language. Focus on correctness, not on following Python style guide or good variable naming. Don't comment on the provided unit tests, they're fixed and not meant to be changed. Your feedback should be understandable to someone who doesn't see these unit tests. If the solution is already okay, just output \"OK\"."),
                HumanMessagePromptTemplate.from_template("""
Here is my task:
{text}

The function should pass the following tests:
{test_list_0}
{test_list_1}
{test_list_2}

Here is my solution:
{initial_solution}

How can I improve it? Just give be a short feedback, I don't need the improved solution.
            """.strip(), input_variables=["text", "test_list_0", "test_list_1", "test_list_2" "initial_solution"]),
            ])
            refinement_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful Python coding assistant. Human will be giving Python programming tasks paired with one unit test. Your job is to write a function that satisfies the specification in task description and passes the unit test. Your replies should consist purely of the improved solution, without any additional comments. Imporant: Do not include the test case in your solution! Output just the improved solution Your entire output should be ready to be copy-pasted into a Python console and run. Human will be giving you feedback on your solution. You should use this feedback to improve your solution. Again, your output should consist purely of the improved solution, without any additional comments. Sometimes human's feedback will be just \"OK\". This means that your solution is already correct and you should repeat it verbatim."),
                HumanMessagePromptTemplate.from_template("""
{text}
        
The function should pass the following tests:
{test_list_0}
{test_list_1}
{test_list_2}
                """.strip(), input_variables=["text", "test_list_0", "test_list_1", "test_list_2"]),
                AIMessagePromptTemplate.from_template("{initial_solution}", input_variables=["initial_solution"]),
                HumanMessagePromptTemplate.from_template("{feedback}", input_variables=["feedback"]),
            ])
        else:
            raise KeyError(chain_name)

        # === 1. Initial solution === #

        initial_solution_chain = LLMChain(
            llm=initial_llm,
            prompt=initial_solution_prompt,
            output_key="initial_solution",
        )
        feedback_chain = LLMChain(llm=feedback_llm, prompt=feedback_prompt, output_key="feedback")
        refinement_chain = LLMChain(llm=refinement_llm, prompt=refinement_prompt, output_key="refinement")
        ilf_chain = SequentialChain(
            chains=[initial_solution_chain, feedback_chain, refinement_chain],
            input_variables=["text", "test_list_0", "test_list_1", "test_list_2"],
            output_variables=["initial_solution", "feedback", "refinement"],
        )
        return ilf_chain

    def process(self, chain, example):
        return chain({
            "text": example["text"],
            # HumanMessagePromptTemplate appears to not be able to handle lists,
            # so we need to pass each element separately.
            "test_list_0": example["test_list"][0],
            "test_list_1": example["test_list"][1],
            "test_list_2": example["test_list"][2],

        })

    def evaluate(self, phase: str, outputs: List[Dict]):
        raise NotImplementedError()
