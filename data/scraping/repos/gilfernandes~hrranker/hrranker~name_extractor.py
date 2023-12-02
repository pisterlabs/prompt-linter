from langchain.chains import create_tagging_chain_pydantic

from hrranker.hr_model import NameExtraction

from hrranker.config import cfg
from hrranker.log_init import logger


def extract_name(file_name: str) -> str:
    logger.info("extract_name: %s %s", file_name, type(file_name))
    chain = create_tagging_chain_pydantic(NameExtraction, cfg.llm)
    name_extraction: NameExtraction = chain.run(file_name)
    return " ".join(name_extraction.person_full_name)


if __name__ == "__main__":
    for pdf in cfg.test_doc_location.glob("*.pdf"):
        file_name = pdf.stem
        logger.info(f"Processing {file_name}")
        name = extract_name(file_name)
        logger.info(f"name {name}")
