import logging
import pathlib
from typing import Dict, Final, cast

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore

from src.domain.paper_format_dto import FormatOchiaiDTO
from src.usecase.summarizer import BaseSummarizer

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OchiaiFormatSummarizer(BaseSummarizer):
    """Summarizer for Ochiai format.

    Args:
        llm_model (BaseLanguageModel): language model to use
        vectorstore (dict[str, VectorStore]): vectorstore to use
        prompt_template_dir_path (pathlib.Path): path to prompt template directory

    """

    def __init__(
        self,
        llm_model: BaseLanguageModel,
        vectorstore: dict[str, VectorStore],
        prompt_template_dir_path: pathlib.Path,
    ) -> None:
        super().__init__(
            llm_model=llm_model,
            vectorstore=vectorstore,
            prompt_template_dir_path=prompt_template_dir_path,
        )

    def summarize(self, verbose: bool = True) -> FormatOchiaiDTO:
        """Summarize paper in Ochiai format.

        Args:
            verbose (bool, optional): whether to print logs. Defaults to True.

        Returns:
            FormatOchiaiDTO: summarized paper in Ochiai format

        """
        outline = self._summarize_outline(verbose=verbose)
        contribution = self._summarize_contribution(verbose=verbose)
        method = self._summarize_method(verbose=verbose)
        evaluation = self._summarize_evaluation(verbose=verbose)
        discussion = self._summarize_discussion(verbose=verbose)
        return FormatOchiaiDTO(
            outline=outline,
            contribution=contribution,
            method=method,
            evaluation=evaluation,
            discussion=discussion,
        )

    def _summarize_outline(self, verbose: bool = True) -> str:
        """`どんなもの？`"""
        prompt_template: Final = self.template_env.get_template(
            "outline_ja.jinja2"
        ).render()
        outline_prompt = PromptTemplate(
            template=prompt_template, input_variables=["text"]
        )
        outline_chain = LLMChain(
            llm=self.llm_model, prompt=outline_prompt, verbose=verbose
        )
        combine_document_chain = StuffDocumentsChain(
            llm_chain=outline_chain,
            document_variable_name="text",
            verbose=verbose,
        )

        retriever = self.vectorstore["wo_abstract"].as_retriever(
            serch_type="similarity",
            search_kwargs={"k": 1},
        )

        selected_documents = []
        # Get top-k relevant documents
        proposed_method: Final = retriever.get_relevant_documents("Proposed method")
        experiments: Final = retriever.get_relevant_documents("Experiments")
        resutls: Final = retriever.get_relevant_documents("Results")

        abstract_document = self._get_abstract_from_vectorstore()
        selected_documents.append(abstract_document)
        selected_documents.extend(proposed_method)
        selected_documents.extend(experiments)
        selected_documents.extend(resutls)
        return cast(str, combine_document_chain.run(selected_documents))

    def _summarize_contribution(self, verbose: bool = True) -> str:
        """`先行研究と比べてどこがすごい？`"""
        contribution_query: Final = "The contribution of this study"
        problem_query: Final = "The problems with previous studies"
        contribution = self._run_combine_document_chain(
            query=contribution_query,
            prompt_template_filename="contribution_ja.jinja2",
            prompt_input_variable="contribution_text",
            verbose=verbose,
        )
        problem = self._run_combine_document_chain(
            query=problem_query,
            prompt_template_filename="problem_ja.jinja2",
            prompt_input_variable="problem_text",
            verbose=verbose,
        )

        combine_template: Final = self.template_env.get_template(
            "combination_ja.jinja2"
        ).render()
        overall_prompt = PromptTemplate(
            input_variables=["contribution", "problem"],
            template=combine_template,
        )
        overall_chain = LLMChain(
            llm=self.llm_model, prompt=overall_prompt, verbose=verbose
        )
        return cast(
            str,
            overall_chain.run(
                {
                    "contribution": contribution,
                    "problem": problem,
                }
            ),
        )

    def _summarize_method(self, verbose: bool = True) -> str:
        """`技術や手法のキモはどこ？`"""
        query: Final = "The proposed method and dataset in this study"

        return self._run_combine_document_chain(
            query=query,
            prompt_template_filename="method_ja.jinja2",
            prompt_input_variable="text",
            verbose=verbose,
        )

    def _summarize_evaluation(self, verbose: bool = True) -> str:
        """`どうやって有効だと検証した？`"""
        query: Final = "The experiments conducted in this study and their evaluation"

        return self._run_combine_document_chain(
            query=query,
            prompt_template_filename="evaluation_ja.jinja2",
            prompt_input_variable="text",
            verbose=verbose,
        )

    def _summarize_discussion(self, verbose: bool = True) -> str:
        """`議論はある？`"""
        query: Final = "The authors' analysis and future prospects based on the results of the evaluation of this study"

        return self._run_combine_document_chain(
            query=query,
            prompt_template_filename="discussion_ja.jinja2",
            prompt_input_variable="text",
            verbose=verbose,
        )

    def _run_combine_document_chain(
        self,
        query: str,
        prompt_template_filename: str,
        prompt_input_variable: str,
        search_type: str = "similarity",
        search_kwargs: Dict = {"k": 5},
        verbose: bool = True,
    ) -> str:
        """Run combine document chain.

        Args:
            query (str): query for vectorstore
            prompt_template_filename (str): template filename for prompt
            prompt_input_variable (str): input variable name for prompt
            search_type (str, optional): search type for vectorstore. Defaults to "similarity".
            search_kwargs (Dict, optional): search kwargs for vectorstore. Defaults to {"k": 5}.
            verbose (bool, optional): verbose. Defaults to True.

        Returns:
            str: summarized text
        """
        prompt_template: Final = self.template_env.get_template(
            prompt_template_filename
        ).render()
        prompt: Final = PromptTemplate(
            template=prompt_template, input_variables=[prompt_input_variable]
        )

        chain: Final = LLMChain(llm=self.llm_model, prompt=prompt, verbose=verbose)
        combine_document_chain: Final = StuffDocumentsChain(
            llm_chain=chain,
            document_variable_name=prompt_input_variable,
            verbose=verbose,
        )

        retriever = self.vectorstore["all"].as_retriever(
            serch_type=search_type,
            search_kwargs=search_kwargs,
        )
        result: Final = retriever.get_relevant_documents(query)
        return cast(str, combine_document_chain.run(result))

    def _get_abstract_from_vectorstore(self) -> Document:
        """Get abstract from vectorstore.

        Returns:
            Document: abstract document
        """
        if self.vectorstore["all"].__class__.__name__ == "FAISS":
            abstract_docstore_id = self.vectorstore["all"].index_to_docstore_id[0]  # type: ignore
            abstract_document = self.vectorstore["all"].docstore.search(abstract_docstore_id)  # type: ignore
        else:
            raise NotImplementedError(
                "Implement the logic to get abstract texts for other vectorstores."
            )
        return abstract_document  # type: ignore
