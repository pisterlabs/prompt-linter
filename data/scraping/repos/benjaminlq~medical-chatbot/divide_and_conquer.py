"""QAOverDocsExperiment Module
"""
import json
import os.path as osp
from abc import abstractmethod
from typing import List, Optional, Union

import pandas as pd
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate

from config import LOGGER, MAIN_DIR
from custom_chain import ReduceDocumentsChainV2
from custom_parsers import DrugOutput, DrugParser

from exp.base import BaseExperiment


class DivideAndConquerBaseExperiment(BaseExperiment):
    """DivideAndConquerBaseExperiment Base Experiment Module"""

    def __init__(
        self,
        keys_json: str = osp.join(MAIN_DIR, "auth", "api_keys.json"),
        temperature: float = 0,
        gt: Optional[str] = None,
        verbose: bool = False,
    ):
        """DivideAndConquerBaseExperiment Base Experiment Module

        Args:
            keys_json (str, optional): Path to API Keys. Defaults to osp.join(MAIN_DIR, "auth", "api_keys.json").
            temperature (float, optional): Temperature Settings for LLM model. Defaults to 0.
            gt (Optional[str], optional): Path to Ground Truth file. Defaults to None.
            verbose (bool, optional): Verbose Setting. Defaults to False.
        """
        super(DivideAndConquerBaseExperiment, self).__init__(
            keys_json=keys_json,
            temperature=temperature,
            gt=gt,
            verbose=verbose,
        )

        self.questions = []
        self.answers = []
        self.intermediate_steps = []
        self.prompt_map = {}

        self.drug_parser = DrugParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=DrugOutput),
            llm=ChatOpenAI(
                model_name="gpt-4", temperature=0, openai_api_key=self.openai_key
            ),
        )

    def run_test_cases(
        self,
        test_cases: Union[List[str], str],
        docs: List[Document],
        return_intermediate_steps: bool = False,
    ):
        """Run and save test cases to memory

        Args:
            test_cases (Union[List[str], str]): List of test queries.
            docs (List[Document]): List of input documents
            return_intermediate_steps (bool, optional): Return intermediate steps. Defaults to False.
        """

        if isinstance(test_cases, str):
            with open(test_cases, "r", encoding="utf-8-sig") as f:
                test_cases = f.readlines()
            test_cases = [test_case.rstrip() for test_case in test_cases]

        if not self.chain:
            self._create_chain(return_intermediate_steps=return_intermediate_steps)

        if self.chain._chain_type == "map_reduce_documents_chain":
            no_tokens = sum(
                [
                    self.chain.combine_document_chain.llm_chain.llm.get_num_tokens(
                        doc.page_content
                    )
                    for doc in docs
                ]
            )
            LOGGER.info(f"Number of tokens from all documents: {no_tokens}")

        for test_case in test_cases:
            print("Query: {}".format(test_case))
            output = self.chain({"input_documents": docs, "question": test_case})
            self.questions.append(output["question"])
            self.answers.append(output["output_text"])
            self.intermediate_steps.append(
                output["intermediate_steps"] if return_intermediate_steps else None
            )

    def reset(self):
        """Reset queries and answers"""
        self.questions = []
        self.answers = []
        self.intermediate_steps = []

    @abstractmethod
    def _create_chain(
        self, return_intermediate_steps: bool = False
    ) -> BaseCombineDocumentsChain:
        """Initiate Main Chain for QA

        Args:
            return_intermediate_steps (bool, optional): Return Intermediate Steps. Defaults to False.

        Returns:
            BaseCombineDocumentsChain: QA Chain
        """
        return NotImplementedError

    def save_json(self, output_path: str):
        """Save Output of test case runs to json file

        Args:
            output_path (str): Output Path to json file.
        """
        output_dict = {
            "prompt": {
                prompt_type: BaseExperiment.convert_prompt_to_string(prompt)
                for prompt_type, prompt in self.prompt_map.items()
            },
            "test_cases": [],
        }

        for question, answer, intermediate_steps in zip(
            self.questions, self.answers, self.intermediate_steps
        ):
            output_dict["test_cases"].append(
                {
                    "question": question,
                    "answer": answer,
                    "intermediate_steps": intermediate_steps,
                }
            )

        with open(output_path, "w") as f:
            json.dump(output_dict, f)

    def load_json(self, json_path: str, reset: bool = False):
        """Load Queries and Answers from Json file

        Args:
            json_path (str): Path to json output file to load into instance
            reset (bool, optional): If reset, clear queries and answers from memory before loading. Defaults to False.
        """
        if reset:
            self.reset()
        with open(json_path, "r") as f:
            input_dict = json.load(f)
        for test_case in input_dict["test_cases"]:
            self.questions.append(test_case["question"])
            self.answers.append(test_case["answer"])
            self.intermediate_steps.append(test_case["intermediate_steps"])
        LOGGER.info("Json file loaded successfully into Experiment instance.")

    def write_csv(self, output_csv: str):
        """Write Output to csv file

        Args:
            output_csv (str): Path to csv file
        """
        info = {"question": self.questions}

        if self.ground_truth is not None:
            info["gt_rec1"] = self.ground_truth["Recommendation 1"].tolist()
            info["gt_rec2"] = self.ground_truth["Recommendation 2"].tolist()
            info["gt_rec3"] = self.ground_truth["Recommendation 3"].tolist()
            info["gt_avoid"] = self.ground_truth["Drug Avoid"].tolist()
            info["gt_reason"] = self.ground_truth["Reasoning"].tolist()

        for prompt_type, prompt in self.prompt_map.items():
            info[f"{prompt_type}_prompt"] = [
                BaseExperiment.convert_prompt_to_string(prompt)
            ] * len(self.questions)

        pd_answers = [[], []]
        pd_pros = [[], []]
        pd_cons = [[], []]
        for answer in self.answers:
            if answer:
                try:
                    drugs = self.drug_parser.parse(answer)
                except Exception:
                    raise Exception("Cannot parse answer properly.")
            else:
                drugs = []

            pd_answers[0].append(drugs[0].drug_name if len(drugs) > 0 else None)
            pd_answers[1].append(drugs[1].drug_name if len(drugs) > 1 else None)
            pd_pros[0].append(drugs[0].advantages if len(drugs) > 0 else None)
            pd_cons[0].append(drugs[0].disadvantages if len(drugs) > 0 else None)
            pd_pros[1].append(drugs[1].advantages if len(drugs) > 1 else None)
            pd_cons[1].append(drugs[1].disadvantages if len(drugs) > 1 else None)

        info["raw_answer"] = self.answers
        info["answer1"] = pd_answers[0]
        info["pro1"] = pd_pros[0]
        info["cons1"] = pd_cons[0]
        info["answer2"] = pd_answers[1]
        info["pro2"] = pd_pros[1]
        info["cons2"] = pd_cons[1]

        panda_df = pd.DataFrame(info)

        panda_df.to_csv(output_csv, header=True)


class MapReduceQAOverDocsExperiment(DivideAndConquerBaseExperiment):
    """MapReduce QA Experiment"""

    def __init__(
        self,
        map_prompt: Optional[Union[PromptTemplate, ChatPromptTemplate]],
        combine_prompt: Optional[Union[PromptTemplate, ChatPromptTemplate]],
        collapse_prompt: Optional[Union[PromptTemplate, ChatPromptTemplate]] = None,
        llm_type: str = "gpt-3.5-turbo",
        reduce_llm: Optional[str] = None,
        collapse_llm: Optional[str] = None,
        keys_json: str = osp.join(MAIN_DIR, "auth", "api_keys.json"),
        temperature: float = 0,
        max_gen_tokens: int = 512,
        combine_max_gen_tokens: int = 512,
        collapse_max_gen_tokens: int = 512,
        combine_max_doc_tokens: int = 14000,
        collapse_max_doc_tokens: int = 6000,
        gt: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """MapReduce QA Experiment

        Args:
            map_prompt (Optional[Union[PromptTemplate, ChatPromptTemplate]]): Prompt to perform QA on each document
            combine_prompt (Optional[Union[PromptTemplate, ChatPromptTemplate]]): Prompt to combine all output for final answer
            collapse_prompt (Optional[Union[PromptTemplate, ChatPromptTemplate]], optional): Prompt to answer on small groups of documents. Defaults to None.
            llm_type (str, optional): Base LLM. By default use for map chain. Defaults to "gpt-3.5-turbo".
            reduce_llm (Optional[str], optional): LLM use for final combine step. Defaults to None.
            collapse_llm (Optional[str], optional): LLM use for collapse step. Defaults to None.
            keys_json (str, optional): Path to API keys. Defaults to osp.join(MAIN_DIR, "auth", "api_keys.json").
            temperature (float, optional): Temperature of all LLM models. Defaults to 0.
            max_gen_tokens (int, optional): Max allowed tokens generated for base LLM. Defaults to 512.
            combine_max_gen_tokens (int, optional): Max allowed tokens generated for combine LLM.. Defaults to 512.
            collapse_max_gen_tokens (int, optional): Max allowed tokens generated for collapse LLM.. Defaults to 512.
            combine_max_doc_tokens (int, optional): Maximum combined tokens at final reduce step. Defaults to 14000.
            collapse_max_doc_tokens (int, optional): Maximum number of tokens allowed for each collapse step. Defaults to 6000.
            gt (Optional[str], optional): Path to ground_truth file. Defaults to None.
            verbose (bool, optional): Verbose settings. Defaults to False.
        """

        super(MapReduceQAOverDocsExperiment, self).__init__(
            keys_json=keys_json,
            temperature=temperature,
            gt=gt,
            verbose=verbose,
        )

        if isinstance(map_prompt, ChatPromptTemplate):
            self.llm = ChatOpenAI(
                model_name=llm_type,
                temperature=self.temperature,
                max_tokens=max_gen_tokens,
                openai_api_key=self.openai_key,
            )
        elif isinstance(map_prompt, PromptTemplate):
            self.llm = OpenAI(
                model_name=llm_type,
                temperature=self.temperature,
                max_tokens=max_gen_tokens,
                openai_api_key=self.openai_key,
            )
        else:
            raise ValueError("Incorrect Type of Map Prompt Template")

        reduce_llm = reduce_llm or llm_type
        collapse_llm = collapse_llm or reduce_llm

        if isinstance(combine_prompt, ChatPromptTemplate):
            self.reduce_llm = ChatOpenAI(
                model_name=reduce_llm,
                temperature=self.temperature,
                max_tokens=combine_max_gen_tokens,
                openai_api_key=self.openai_key,
            )

        elif isinstance(combine_prompt, PromptTemplate):
            self.reduce_llm = OpenAI(
                model_name=reduce_llm,
                temperature=self.temperature,
                max_tokens=combine_max_gen_tokens,
                openai_api_key=self.openai_key,
            )
        else:
            raise ValueError("Incorrect Type of Combine Prompt Template")

        self.map_prompt = map_prompt
        self.combine_prompt = combine_prompt
        self.collapse_prompt = collapse_prompt or combine_prompt

        self.prompt_map = {
            "map": map_prompt,
            "combine": combine_prompt,
            "collapse": collapse_prompt,
        }

        if isinstance(self.collapse_prompt, ChatPromptTemplate):
            self.collapse_llm = ChatOpenAI(
                model_name=collapse_llm,
                temperature=self.temperature,
                max_tokens=collapse_max_gen_tokens,
                openai_api_key=self.openai_key,
            )

        elif isinstance(self.collapse_prompt, PromptTemplate):
            self.collapse_llm = OpenAI(
                model_name=collapse_llm,
                temperature=self.temperature,
                max_tokens=collapse_max_gen_tokens,
                openai_api_key=self.openai_key,
            )
        else:
            raise ValueError("Incorrect Type of Collapse Prompt Template")

        self.combine_max_doc_tokens = combine_max_doc_tokens
        self.collapse_max_doc_tokens = collapse_max_doc_tokens

    def _create_chain(self, return_intermediate_steps: bool = False):
        """Initiate QA from Source Chain

        Args:
            return_intermediate_steps (bool, optional): Whether to return intermediate_steps. Defaults to True.
        """

        map_chain = LLMChain(llm=self.llm, prompt=self.map_prompt, verbose=self.verbose)

        reduce_chain = LLMChain(
            llm=self.reduce_llm, prompt=self.combine_prompt, verbose=self.verbose
        )

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="summaries",
            verbose=self.verbose,
        )

        collapse_chain = LLMChain(
            llm=self.collapse_llm, prompt=self.collapse_prompt, verbose=self.verbose
        )

        collapse_documents_chain = StuffDocumentsChain(
            llm_chain=collapse_chain,
            document_variable_name="summaries",
            verbose=self.verbose,
        )
        
        reduce_document_chain = ReduceDocumentsChainV2(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=collapse_documents_chain,
            combine_max_tokens=self.combine_max_doc_tokens,
            collapse_max_tokens=self.collapse_max_doc_tokens
        )

        self.chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_document_chain,
            document_variable_name="context",
            return_intermediate_steps=return_intermediate_steps,
            return_map_steps=return_intermediate_steps,
            verbose=self.verbose,
        )


class RefineQAOverDocsExperiment(DivideAndConquerBaseExperiment):
    """Refine QA Experiment"""

    def __init__(
        self,
        initial_prompt: Optional[Union[PromptTemplate, ChatPromptTemplate]],
        refine_prompt: Optional[Union[PromptTemplate, ChatPromptTemplate]],
        llm_type: str = "gpt-3.5-turbo",
        refine_llm: Optional[str] = None,
        keys_json: str = osp.join(MAIN_DIR, "auth", "api_keys.json"),
        temperature: float = 0,
        max_gen_tokens: int = 1024,
        gt: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Refine QA Experiment

        Args:
            initial_prompt (Optional[Union[PromptTemplate, ChatPromptTemplate]]): Prompt for initial answer
            refine_prompt (Optional[Union[PromptTemplate, ChatPromptTemplate]]): Prompt for each refine step.
            llm_type (str, optional): Base LLM. By default use for initial & refine chain. Defaults to "gpt-3.5-turbo".
            refine_llm (Optional[str], optional): LLM for refine chain. In None, assign to base llm type. Defaults to None.
            keys_json (str, optional): Path to API keys file. Defaults to osp.join(MAIN_DIR, "auth", "api_keys.json").
            temperature (float, optional): Temperature of all LLM models. Defaults to 0.
            max_gen_tokens (int, optional): Maximum generated tokens allowed. Defaults to 1024.
            gt (Optional[str], optional): Path to ground truth file. Defaults to None.
            verbose (bool, optional): Verbose settings. Defaults to False.
        """

        super(RefineQAOverDocsExperiment, self).__init__(
            keys_json=keys_json,
            temperature=temperature,
            gt=gt,
            verbose=verbose,
        )

        if isinstance(initial_prompt, ChatPromptTemplate):
            self.llm = ChatOpenAI(
                model_name=llm_type,
                temperature=self.temperature,
                max_tokens=max_gen_tokens,
                openai_api_key=self.openai_key,
            )
        elif isinstance(initial_prompt, PromptTemplate):
            self.llm = OpenAI(
                model_name=llm_type,
                temperature=self.temperature,
                max_tokens=max_gen_tokens,
                openai_api_key=self.openai_key,
            )
        else:
            raise ValueError("Incorrect Type of Map Prompt Template")

        refine_llm = refine_llm or llm_type

        if isinstance(refine_prompt, ChatPromptTemplate):
            self.refine_llm = ChatOpenAI(
                model_name=refine_llm,
                temperature=self.temperature,
                max_tokens=max_gen_tokens,
                openai_api_key=self.openai_key,
            )

        elif isinstance(refine_prompt, PromptTemplate):
            self.refine_llm = OpenAI(
                model_name=refine_llm,
                temperature=self.temperature,
                max_tokens=max_gen_tokens,
                openai_api_key=self.openai_key,
            )

        else:
            raise ValueError("Incorrect Type of Combine Prompt Template")

        self.initial_prompt = initial_prompt
        self.refine_prompt = refine_prompt

        self.prompt_map = {
            "initial": initial_prompt,
            "refine": refine_prompt,
        }

    def _create_chain(self, return_intermediate_steps: bool = False):
        """Initiate QA from Source Chain

        Args:
            return_intermediate_steps (bool, optional): Whether to return intermediate_steps. Defaults to True.
        """

        initial_chain = LLMChain(
            llm=self.llm, prompt=self.initial_prompt, verbose=self.verbose
        )

        refine_chain = LLMChain(
            llm=self.refine_llm, prompt=self.refine_prompt, verbose=self.verbose
        )

        self.chain = RefineDocumentsChain(
            initial_llm_chain=initial_chain,
            refine_llm_chain=refine_chain,
            document_variable_name="context_str",
            initial_response_name="existing_answer",
            return_intermediate_steps=return_intermediate_steps,
            verbose=self.verbose,
        )
