"""Experiment Module
"""
import json
import os
import os.path as osp
import re
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
import pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder

from config import LOGGER, MAIN_DIR
from custom_parsers import DrugOutput
from models.splade import SpladeEncoder
from utils import convert_csv_to_documents, load_documents

from .base import BaseExperiment


class QAWithPineconeHybridSearchExperiment(BaseExperiment):
    def __init__(
        self,
        prompt_template: Union[PromptTemplate, ChatPromptTemplate],
        sparse_model_path: str,
        pinecone_idx_name: str,
        sparse_type: Literal["splade", "bm25"] = "splade",
        llm_type: str = "gpt-3.5-turbo",
        emb: str = "text-embedding-ada-002",
        keys_json: str = osp.join(MAIN_DIR, "auth", "api_keys.json"),
        temperature: float = 0,
        max_tokens: int = 512,
        gt: Optional[str] = None,
        verbose: bool = False,
        k: int = 4,
        alpha: float = 0.5,
        max_tokens_limit: int = 3375,
        reduce_k_below_max_tokens: bool = True,
        device: str = "cpu",
    ):

        super(QAWithPineconeHybridSearchExperiment, self).__init__(
            llm_type=llm_type,
            keys_json=keys_json,
            temperature=temperature,
            max_tokens=max_tokens,
            gt=gt,
            verbose=verbose,
        )

        ## Initialize Pinecone session
        self.pinecone_idx_name = pinecone_idx_name
        self.pinecone_api_key = self.keys[f"PINECONE_API_{sparse_type.upper()}"]["KEY"]
        self.pinecone_env = self.keys[f"PINECONE_API_{sparse_type.upper()}"]["ENV"]

        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)

        if pinecone_idx_name not in pinecone.list_indexes():
            Warning("Index Name does not exist")
            pinecone.create_index(
                name=pinecone_idx_name,
                dimension=1536,
                metric="dotproduct",
                pod_type="s1",
                metadata_config={"indexed": []},
            )

        self.index = pinecone.Index(pinecone_idx_name)

        if isinstance(prompt_template, ChatPromptTemplate):
            self.llm = ChatOpenAI(
                model_name=self.llm_type,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.openai_key,
            )

        else:
            self.llm = OpenAI(
                model_name=self.llm_type,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.openai_key,
            )

        self.dense_embeddings = OpenAIEmbeddings(
            model=emb, openai_api_key=self.openai_key
        )
        self.sparse_type = sparse_type
        self.sparse_model_path = sparse_model_path
        self.device = device
        self.k = k
        assert 0 <= alpha <= 1, "Invalid alpha"
        self.alpha = alpha
        self.max_tokens_limit = max_tokens_limit
        self.reduce_k_below_max_tokens = reduce_k_below_max_tokens
        self.prompt_template = prompt_template

        try:
            self.load_vectorstore(
                sparse_model_path=self.sparse_model_path,
                sparse_type=self.sparse_type,
                embeddings=self.dense_embeddings,
            )
        except Exception:
            raise Exception(
                "Vectorstore invalid. Please load valid vectorstore or create new vectorstore."
            )

        self.questions = []
        self.answers = []
        self.sources = []
        self.drug_parser = PydanticOutputParser(pydantic_object=DrugOutput)

    def load_vectorstore(
        self,
        sparse_model_path: str,
        sparse_type: Literal["splade", "bm25"],
        embeddings: Embeddings,
    ):
        if sparse_type == "bm25":
            sparse_model = BM25Encoder().load(sparse_model_path)
            print("Using BM25 Sparse Model")
        elif sparse_type == "splade":
            sparse_model = SpladeEncoder(
                model_path=sparse_model_path,
                max_seq_length=512,
                agg="max",
                device=self.device,
            )
            print("Using Term Expansion SPLADE Sparse Model")
        else:
            raise ValueError(f"Sparse Embedding of type {sparse_type} does not exist")

        self.retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=sparse_model,
            index=self.index,
            top_k=self.k,
            alpha=self.alpha,
        )

        LOGGER.info("Successfully loaded existing vectorstore from local storage")

    def generate_vectorstore(
        self,
        source_directory: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 250,
        exclude_pages: Optional[Dict] = None,
        additional_docs: Optional[str] = None,
    ):

        if source_directory:
            LOGGER.info(f"Loading documents from {source_directory}")

            documents = load_documents(source_directory, exclude_pages=exclude_pages)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            texts = text_splitter.split_documents(documents)

            LOGGER.info(f"Loaded {len(documents)} documents from {source_directory}")
            LOGGER.info(
                f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)"
            )
        else:
            texts = []

        if additional_docs:
            with open(additional_docs, "r") as f:
                add_doc_infos = json.load(f)
            for add_doc_info in add_doc_infos:
                if add_doc_info["mode"] == "table":
                    texts.extend(convert_csv_to_documents(add_doc_info))
                else:
                    LOGGER.warning(
                        "Invalid document type. No texts added to documents list"
                    )

        page_contents = [text.page_content for text in texts]
        metadatas = [text.metadata for text in texts]

        self.retriever.add_texts(texts=page_contents, metadatas=metadatas)

    def run_test_cases(
        self,
        test_cases: Union[List[str], str],
        only_return_source: bool = False,
        chain_type: Literal["stuff", "refine", "map_reduce", "map_rerank"] = "stuff",
    ):

        if isinstance(test_cases, str):
            with open(test_cases, "r", encoding="utf-8-sig") as f:
                test_cases = f.readlines()
            test_cases = [test_case.rstrip() for test_case in test_cases]

        if not self.chain:
            self._create_chain(chain_type=chain_type)

        if only_return_source:
            LOGGER.info("Perform Semantic Search for Source Documents only (No QA).")

        for test_case in test_cases:
            print("Query: {}".format(test_case))
            sources = []  # All sources for 1 single query
            if only_return_source:
                self.questions.append(test_case)
                self.answers.append(None)
                inputs = {"question": test_case}
                source_documents = self.chain._get_docs(inputs)

            else:
                output = self.chain(test_case)
                self.questions.append(output["question"])
                self.answers.append(output["answer"])
                source_documents = output["source_documents"]

            token_count = 0
            for document in source_documents:
                sources.append(
                    {
                        "title": document.metadata["title"],
                        "filename": document.metadata["source"].split("/")[-1],
                        "page": document.metadata["page"],
                        "text": document.page_content,
                    }
                )
                token_count += (
                    self.chain.combine_documents_chain.llm_chain.llm.get_num_tokens(
                        document.page_content
                    )
                )

            self.sources.append(sources)
            LOGGER.info(
                f"{token_count} tokens were added to prompt from {len(source_documents)} documents"
            )

    def reset(self):
        """Reset queries and answers"""
        self.questions = []
        self.answers = []
        self.sources = []

    @staticmethod
    def process_source(source: Dict) -> str:
        """Process Source document object

        Args:
            source (Dict): Source Document Information

        Returns:
            str: Source Document Information in string
        """
        return "\n\n".join([f"{k}: {v}" for k, v in source.items()])

    def save_json(self, output_path: str):
        """Save Output of test case runs to json file

        Args:
            output_path (str): Output Path to json file.
        """
        output_dict = {}
        output_dict[
            "prompt"
        ] = QAWithPineconeHybridSearchExperiment.convert_prompt_to_string(
            self.prompt_template
        )
        output_dict["test_cases"] = []

        for question, answer, source in zip(self.questions, self.answers, self.sources):
            output_dict["test_cases"].append(
                {"question": question, "answer": answer, "sources": source}
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
            self.sources.append(test_case["sources"])
        LOGGER.info("Json file loaded successfully into Experiment instance.")

    def write_csv(self, output_csv: str, num_docs: int = 10):
        """Write questions and answers to .csv files

        Args:
            output_csv (str): Path to output csv file
        """

        pd_answers = [[], []]
        pd_pros = [[], []]
        pd_cons = [[], []]
        pd_sources = [[] for _ in range(num_docs)]

        for answer, sources in zip(self.answers, self.sources):
            if answer:
                drugs_info = re.findall(re.compile(r"{[^{}]+}"), answer)
                drugs = []
                for drug in drugs_info:
                    try:
                        drug = self.drug_parser.parse(drug)
                        drugs.append(drug)
                    except Exception:
                        pass
            else:
                drugs = []

            pd_answers[0].append(drugs[0].drug_name if len(drugs) > 0 else None)
            pd_answers[1].append(drugs[1].drug_name if len(drugs) > 1 else None)
            pd_pros[0].append(drugs[0].advantages if len(drugs) > 0 else None)
            pd_cons[0].append(drugs[0].disadvantages if len(drugs) > 0 else None)
            pd_pros[1].append(drugs[1].advantages if len(drugs) > 1 else None)
            pd_cons[1].append(drugs[1].disadvantages if len(drugs) > 1 else None)

            for idx, source in enumerate(sources):
                pd_sources[idx].append(
                    QAWithPineconeHybridSearchExperiment.process_source(source)
                )

            if idx + 1 < len(pd_sources):
                for i in range(idx + 1, len(pd_sources)):
                    pd_sources[i].append(None)

        info = {"question": self.questions}

        if self.ground_truth is not None:
            info["gt_rec1"] = self.ground_truth["Recommendation 1"].tolist()
            info["gt_rec2"] = self.ground_truth["Recommendation 2"].tolist()
            info["gt_rec3"] = self.ground_truth["Recommendation 3"].tolist()
            info["gt_avoid"] = self.ground_truth["Drug Avoid"].tolist()
            info["gt_reason"] = self.ground_truth["Reasoning"].tolist()

        info["prompt"] = [
            QAWithPineconeHybridSearchExperiment.convert_prompt_to_string(
                self.prompt_template
            )
        ] * len(self.questions)
        info["raw_answer"] = self.answers
        info["answer1"] = pd_answers[0]
        info["pro1"] = pd_pros[0]
        info["cons1"] = pd_cons[0]
        info["answer2"] = pd_answers[1]
        info["pro2"] = pd_pros[1]
        info["cons2"] = pd_cons[1]

        for idx, pd_source in enumerate(pd_sources):
            info[f"source{idx+1}"] = pd_source

        panda_df = pd.DataFrame(info)

        panda_df.to_csv(output_csv, header=True)

    def _create_chain(
        self,
        chain_type: Literal["stuff", "map_reduce", "map_rerank", "refine"] = "stuff",
        return_source_documents: bool = True,
    ):
        """Initiate QA from Source Chain

        Args:
            chain_type (str, optional): Chain Type. Can be stuff|map_reduce|refine|map_rerank. Defaults to "stuff".
            return_source_documents (bool, optional): Whether to return source documents along side answers. Defaults to True.
        """
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": self.prompt_template},
            max_tokens_limit=self.max_tokens_limit,
            reduce_k_below_max_tokens=self.reduce_k_below_max_tokens,
            verbose=self.verbose,
        )
