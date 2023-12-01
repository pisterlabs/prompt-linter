"""

This script generates summaries of all CVPR 2023 papers. Each summary is
saved under `data/papers/<family_name>_<paper_title>/` directory. This
script requires following environmental variable.

- OPENAI_API_KEY

"""
import json
import logging
import pathlib
from typing import Final

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS

from src.latex_parser import parse_latex_text, structure_latex_documents
from src.parser import Paper
from src.summarizer import OchiaiFormatPaperSummarizer

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Note: list config
llm_model_name: str = "gpt-3.5-turbo"
temperature: float = 0.9
chunk_size: int = 200
chunk_overlap: int = 40

# Check JSON file existence.
paper_info_path: Final = pathlib.Path("./data/papers.json")
if not paper_info_path.exists():
    error_message: Final = f"This scripts requires `{str(paper_info_path)}`. \
        Please run `parse_cvf_page.py` frist to generate JSON file."
    raise FileNotFoundError(error_message)

# Load JSON and validate by Pydantic model.
paper_root_path: Final = pathlib.Path("./data/papers/")
with paper_info_path.open("r") as f:
    papers: Final = [Paper.parse_obj(p) for p in json.load(f)]


# Loop over all papers.
for i, paper in enumerate(papers):
    # stem is like: <family_name>_<paper_title>_CVPR_2023_paper
    stem = str(pathlib.Path(paper.pdf).stem)
    directory_path = paper_root_path / stem.removesuffix("_CVPR_2023_paper")

    # Check if PDF file exists or not.
    pdf_file_path = directory_path / (stem + ".pdf")
    if not pdf_file_path.exists():
        raise FileNotFoundError(
            f"`{str(pdf_file_path)}` does not exist. Please run `download_papers.py` frist to download PDF file."
        )

    # If there is no mathpix file, send PDF to mathpix API.
    mathpix_file_path = directory_path / (stem + "_mathpix.txt")
    if not mathpix_file_path.exists():
        raise FileNotFoundError(
            f"`{str(mathpix_file_path)}` does not exist. Please run `convert_to_latex.py` frist to get latex format text file."
        )

    # If summary already exists, continue the loop.
    summary_file_path = directory_path / (stem + "_summary.json")
    if summary_file_path.exists():
        logger.info(f"`{str(summary_file_path)}` already exists. Continue the loop.")
        continue

    # Parse Latex format text.
    raw_paper = TextLoader(file_path=str(mathpix_file_path)).load()[0]
    parsed_paper = parse_latex_text(raw_paper.page_content)

    # Convert text into to structured documents
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        model_name=llm_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = structure_latex_documents(
        parsed_paper,
        text_splitter,
        paper.abstract,
    )

    # Embed documents and store into vector database.
    embeddings = OpenAIEmbeddings()
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
    vectorstore.save_local(str(directory_path / "index"))
    vectorstore.save_local(str(directory_path / "index_wo_abstract"))

    # Generate summary.
    llm_model = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
    summarizer = OchiaiFormatPaperSummarizer(
        llm_model=llm_model,
        vectorstore={
            "all": vectorstore,
            "wo_abstract": vectorstore_wo_abstract,
        },
        prompt_template_dir_path=pathlib.Path("./src/prompts"),
    )
    summary = summarizer.summarize()

    # Save summary.
    with summary_file_path.open("w", encoding="utf-8") as f:
        json.dump(summary.dict(), f, indent=4, ensure_ascii=False)
