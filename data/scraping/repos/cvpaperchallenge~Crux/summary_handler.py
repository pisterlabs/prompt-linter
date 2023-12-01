import json
import logging
import pathlib
from typing import Final, cast

from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter, TokenTextSplitter
from langchain.vectorstores import FAISS

from src.domain.paper_format_dto import FORMAT_MAPPING, SummaryConfigDTO, SummaryFormat
from src.domain.parsed_paper_dto import ParsedPaperDTO
from src.usecase.summarizer import BaseSummarizer

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SummaryHandler:
    """Handler for summarization."""

    def __init__(self, summarizer: type[BaseSummarizer]) -> None:
        self.summarizer = summarizer

    def make_summary(
        self,
        parsed_paper: ParsedPaperDTO,
        pdf_file_path: pathlib.Path,
        summary_config: SummaryConfigDTO,
    ) -> SummaryFormat:
        """Make a summary from the parsed pdf.

        Args:
            parsed_paper (ParsedPaperDTO): Parsed paper.
            pdf_file_path (pathlib.Path): Path to the pdf file.
            summary_config (SummaryConfigDTO): Summary configuration.

        Returns:
            SummaryFormat: Summary of the paper in the specific format.
        """
        stem = str(pdf_file_path.stem)
        summary_file_path = pdf_file_path.parent / (stem + "_summary.json")

        # If summary already exists, continue the loop.
        if summary_file_path.exists():
            logger.info(
                f"`{str(summary_file_path)}` already exists. Continue the loop."
            )
            with summary_file_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            summary_format = FORMAT_MAPPING[summary_config.summary_type]
            summary = summary_format.parse_obj(summary)

        else:
            # Convert text into to structured documents
            text_splitter = TokenTextSplitter.from_tiktoken_encoder(
                model_name=summary_config.llm_model_name,
                chunk_size=summary_config.chunk_size,
                chunk_overlap=summary_config.chunk_overlap,
            )
            documents = self.structure_latex_documents(
                parsed_paper,
                text_splitter,
            )

            # Embed documents and store into vector database.
            embeddings = OpenAIEmbeddings()  # type: ignore
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings,
            )
            vectorstore_wo_abstract = FAISS.from_documents(
                documents=[
                    document
                    for document in documents
                    if document.metadata["section"] != "abstract"
                ],
                embedding=embeddings,
            )

            # Save vector database.
            vectorstore.save_local(str(pdf_file_path.parent / "index"))
            vectorstore.save_local(str(pdf_file_path.parent / "index_wo_abstract"))

            # Generate summary.
            llm_model = ChatOpenAI(
                model_name=summary_config.llm_model_name,
                temperature=summary_config.temperature,
            )  # type: ignore
            summarizer = self.summarizer(
                llm_model=llm_model,
                vectorstore={
                    "all": vectorstore,
                    "wo_abstract": vectorstore_wo_abstract,
                },
                prompt_template_dir_path=pathlib.Path("./src/domain/prompts"),
            )
            summary = summarizer.summarize()

            # Save summary.
            with summary_file_path.open("w", encoding="utf-8") as f:
                json.dump(summary.dict(), f, indent=4, ensure_ascii=False)

        return cast(SummaryFormat, summary)

    def structure_latex_documents(
        self,
        parsed_paper: ParsedPaperDTO,
        text_splitter: TextSplitter,
        abstract_text: str | None = None,
    ) -> list[Document]:
        """Convert parsed paper into structured documents.

        Args:
            parsed_paper (ParsedPaperDTO): Parsed paper.
            text_splitter (TextSplitter): Text splitter.
            abstract_text (str | None, optional): Full abstract text. Defaults to None.

        Returns:
            list[Document]: Structured documents.
        """
        # If full abstract is provided, use it instead of parsed one.
        documents = [
            Document(
                page_content=abstract_text if abstract_text else parsed_paper.abstract,
                metadata={"section": "abstract"},
            )
        ]

        # Loop over section.
        for each_section in parsed_paper.section:
            section_title = each_section.section_title
            if section_title == "References" or section_title == "REFERENCES":
                continue

            section_id = each_section.section_id
            section_text = each_section.section_text
            if section_text:
                metadata = {
                    "section_id": f"{section_id}",
                    "section": f"{section_title}",
                }
                for each_section_text in text_splitter.split_text(section_text):
                    documents.append(
                        Document(page_content=each_section_text, metadata=metadata)
                    )

            # Loop over subsection.
            for each_subsection in each_section.subsection_list:
                subsection_title = each_subsection.subsection_title
                subsection_id = each_subsection.subsection_id
                subsection_text = each_subsection.subsection_text
                metadata = {
                    "section_id": f"{section_id}.{subsection_id}",
                    "section": f"{section_title}/{subsection_title}",
                }
                for each_subsection_text in text_splitter.split_text(subsection_text):
                    documents.append(
                        Document(page_content=each_subsection_text, metadata=metadata)
                    )

        return documents
