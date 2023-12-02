import pandas as pd
from typing import List, Dict, Optional
import tqdm.auto as tqdm
import datasets

from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatAnthropic
from langchain.schema import (
    SystemMessage
)

from ...utils.io import read_json
from ...utils.models import get_chat_model
from .base import BaseTask

MAX_NUM_QUERIES = 3


class HotPotQATask(BaseTask):
    """HotPotQA task"""

    def __init__(self, task_args_str):
        from ...utils.contriever import Contriever
        hotpotqa_config = read_json(task_args_str)
        if "passage_path" in hotpotqa_config:
            self.contriever = Contriever.setup(
                passage_path=hotpotqa_config["passage_path"],
                index_path=hotpotqa_config["index_path"],
            )
        else:
            self.contriever = None

    def get_dataset(self, phase: str):
        return datasets.load_dataset("hotpot_qa", "fullwiki")[phase]

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = None):
        # 0. Setup
        assert chain_name is None
        initial_llm = get_chat_model(model_name=generation_llm)
        feedback_llm = get_chat_model(model_name=feedback_llm)
        refinement_llm = get_chat_model(model_name=refinement_llm)

        # === 1a. Initial search === #
        initial_search_terms_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a question that we would like to answer. 

Question: {question}

To help answer this question, output a list of up to 3 search queries that we want to search Google or Wikipedia for, in the following format:
<search>...</search>
<search>...</search>
<search>...</search>
            """.strip(), input_variables=["question"])
        ])
        initial_search_terms_chain = LLMChain(llm=initial_llm, prompt=initial_search_terms_prompt, output_key="initial_search_terms")

        # === 1b. Initial answer === #
        initial_answer_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a question that we would like to answer. 

Question: {question}

To help answer this question, we ran a quick Google/Wikipedia search and obtained the following excerpts:

{formatted_search_result}

Based on the search results, output the answer to the above question.
            """.strip(), input_variables=["question", "formatted_search_result"])
        ])
        initial_answer_chain = LLMChain(llm=initial_llm, prompt=initial_answer_prompt, output_key="initial_answer")

        # === 2. Feedback === #
        ilf_feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a question that we would like to answer. 

Question: {question}

To help answer this question, a student ran a quick Google/Wikipedia search and obtained the following excerpts:

{formatted_search_result}

The student then read the above search results and provided the following answer:
"{initial_answer}"

How would you improve the above search and answer? Please provide feedback on both the choice of search terms as well as the final answers.
            """.strip(), input_variables=["question", "formatted_search_result", "initial_answer"])
        ])
        feedback_chain = LLMChain(llm=feedback_llm, prompt=ilf_feedback_prompt, output_key="feedback")

        # === 3a. Refinement Search === #
        refinement_search_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a question that we would like to answer. 

Question: {question}

A previous student performed a search on the following search terms:
{formatted_search_terms}

Then they provided the following answer:
"{initial_answer}"

A teacher then provided the following feedback:
"{feedback}"

Based on the above, output a list of up to 3 search queries that we want to search Google or Wikipedia for, in the following format:
<search>...</search>
<search>...</search>
<search>...</search>
            """.strip(), input_variables=["question", "formatted_search_terms", "initial_answer", "feedback"])
        ])
        refinement_search_chain = LLMChain(llm=refinement_llm, prompt=refinement_search_prompt,
                                           output_key="refinement_search_terms")

        # === 3b. Refinement answer === #
        refinement_answer_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a question that we would like to answer. 

Question: {question}

A previous student performed a search on the following search terms:
{formatted_search_terms}

Then they provided the following answer:
"{initial_answer}"

A teacher then provided the following feedback:
"{feedback}"

We took the above into account and ran a Google/Wikipedia search and obtained the following excerpts:

{formatted_refinement_search_result}

Based on the search results, output the answer to the above question.
            """.strip(), input_variables=["question", "formatted_search_terms", "initial_answer", "feedback",
                                          "formatted_refinement_search_result"])
        ])
        refinement_answer_chain = LLMChain(llm=refinement_llm, prompt=refinement_answer_prompt,
                                           output_key="refinement_answer")
        return {
            "initial_search_chain": initial_search_terms_chain,
            "initial_answer_chain": initial_answer_chain,
            "feedback_chain": feedback_chain,
            "refinement_search_chain": refinement_search_chain,
            "refinement_answer_chain": refinement_answer_chain,
        }

    def process(self, chain, example):
        return self.batch_process(chain, [example])[0]

    def batch_process(self, chain, example_list):
        num_examples = len(example_list)

        # Get initial search terms
        search_terms_list = []
        out1_list = []
        for example in tqdm.tqdm(example_list, desc="Initial"):
            out1 = chain["initial_search_chain"]({"question": example["question"]})
            out1_list.append(out1)
            search_terms = parse_search_terms(out1["initial_search_terms"])[:MAX_NUM_QUERIES]
            search_terms_list.append(search_terms)

        search_result_list = self.contriever.get_multi_passages_batched(search_terms_list, top_k=2)

        refinement_search_terms_list = []
        out2_list, out3_list, out4_list = [], [], []
        formatted_search_terms_list = []
        for i in tqdm.trange(num_examples, desc="Process Initial and Refine"):
            example = example_list[i]
            search_terms = search_terms_list[i]
            search_result = search_result_list[i]

            formatted_search_result = format_search_results(search_result)

            out2 = chain["initial_answer_chain"]({
                "question": example["question"],
                "formatted_search_result": formatted_search_result,
            })
            out2_list.append(out2)
            out3 = chain["feedback_chain"]({
                "question": example["question"],
                "formatted_search_result": formatted_search_result,
                "initial_answer": out2["initial_answer"],
            })
            out3_list.append(out3)
            formatted_search_terms = "\n".join(f"- {search_term}" for search_term in search_terms)
            formatted_search_terms_list.append(formatted_search_terms)
            out4 = chain["refinement_search_chain"]({
                "question": example["question"],
                "formatted_search_terms": formatted_search_terms,
                "initial_answer": out2["initial_answer"],
                "feedback": out3["feedback"],
            })
            out4_list.append(out4)
            refinement_search_terms = parse_search_terms(out4["refinement_search_terms"])[:MAX_NUM_QUERIES]
            refinement_search_terms_list.append(refinement_search_terms)

        refinement_search_result_list = self.contriever.get_multi_passages_batched(
            refinement_search_terms_list, top_k=2)

        out_list = []
        for i in tqdm.trange(num_examples, desc="Process Refinement"):
            example = example_list[i]
            refinement_search_result = refinement_search_result_list[i]
            search_result = search_result_list[i]
            out1, out2, out3, out4 = out1_list[i], out2_list[i], out3_list[i], out4_list[i]
            formatted_search_terms = formatted_search_terms_list[i]

            formatted_refinement_search_result = format_search_results(refinement_search_result)
            out5 = chain["refinement_answer_chain"]({
                "question": example["question"],
                "formatted_search_terms": formatted_search_terms,
                "initial_answer": out2["initial_answer"],
                "feedback": out3["feedback"],
                "formatted_refinement_search_result": formatted_refinement_search_result,
            })
            out = {
                "question": example["question"],
                "initial_search_terms": out1["initial_search_terms"],
                "search_results": search_result,
                "initial_answer": out2["initial_answer"],
                "feedback": out3["feedback"],
                "refinement_search_terms": out4["refinement_search_terms"],
                "refinement_search_result": refinement_search_result,
                "refinement_answer": out5["refinement_answer"],
            }
            out_list.append(out)
        return out_list

    def evaluate(self, phase: str, outputs: List[Dict]):
        dataset = self.get_dataset(phase=phase)
        scores = {"initial_score": [], "refined_score": [], "initial_raw": [], "refined_raw": []}
        for row, example in zip(tqdm.tqdm(outputs), dataset):
            initial_judgment = self.score_single(example=example, answer=row["initial_answer"])
            refined_judgment = self.score_single(example=example, answer=row["refinement_answer"])
            scores["initial_score"].append(initial_judgment["judgment"])
            scores["refined_score"].append(refined_judgment["judgment"])
            scores["initial_raw"].append(initial_judgment["raw_judgment"])
            scores["refined_raw"].append(refined_judgment["raw_judgment"])
        return {
            "initial_score": float(pd.Series(scores["initial_score"]).mean()),
            "refined_score": float(pd.Series(scores["refined_score"]).mean()),
            "initial_raw": scores["initial_raw"],
            "refined_raw": scores["initial_raw"],
        }

    @classmethod
    def score_single(cls, example, answer):
        llm = ChatAnthropic(model="claude-instant-v1.1")
        judge_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a homework grading assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a question on a quiz, where the student is allowed to look up information.

QUESTION: {question}

The answer key states that the answer to the question is the following:

ANSWER: {true_answer}

A student wrote the following answer:

Student's Answer: {answer}


Think step-by-step, whether the student answered the question correctly, based on the answer key.
Then, output "CORRECT" is the answer is correct, and "WRONG" otherwise.
It is okay if the student provides more information than necessary. However, if the student is unable to answer, that counts as being wrong.
Your output should look like:
<reasoning> ... </reasoning>
<score> CORRECT / WRONG </score>
                    """.strip(), input_variables=["question", "true_answer", "answer"])
        ])
        chain = LLMChain(llm=llm, prompt=judge_prompt,
                         output_key="judgment")
        raw_judgment = chain({
            "question": example["question"],
            "true_answer": example["answer"],
            "answer": answer.strip(),
        })["judgment"]
        return {"raw_judgment": raw_judgment, "judgment": True if "CORRECT" in raw_judgment else False}


def parse_search_terms(string):
    out_list = []
    parts = string.split("<search>")
    for part in parts:
        if not part.strip():
            continue
        out_list.append(part.split("</search>")[0])
    return out_list


def format_search_results(search_result):
    lines = []
    for i, (query, passages) in enumerate(search_result.items()):
        lines.append(f"Search {i+1}: \"{query}\"")
        for passage in passages:
            lines.append(f"    Article: {passage['title']}")
            lines.append(f"    Excerpt: {passage['text']}")
            lines.append("")
        lines.append("")
        lines.append("===")
        lines.append("")
    return "\n".join(lines)
